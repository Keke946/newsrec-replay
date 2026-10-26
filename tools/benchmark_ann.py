"""Fixed Faiss snapshot benchmark on an existing two-tower; no model fitting."""
import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
from newsrec.data import file_hash,source_hash,write_json


def benchmark(index,queries,timestamps,k,ef_search=None):
    import numpy as np
    index.search(queries[:10],timestamps[:10],k,ef_search)
    batch_times=[]
    for _ in range(3):
        clock=time.perf_counter()
        scores,ids=index.search(queries,timestamps,k,ef_search)
        batch_times.append((time.perf_counter()-clock)*1000)
    single=[]
    for position in range(len(queries)):
        clock=time.perf_counter()
        index.search(queries[position:position+1],timestamps[position:position+1],k,ef_search)
        single.append((time.perf_counter()-clock)*1000)
    return scores,ids,dict(batch_ms=batch_times,batch_median_ms=float(np.median(batch_times)),
                           single_query_ms=dict(p50=float(np.median(single)),p95=float(np.quantile(single,.95)),
                                                mean=float(np.mean(single))))


def main():
    import faiss
    import numpy as np
    import torch
    from newsrec.two_tower import VectorStore,TwoTower,build_examples
    from newsrec.ann import SnapshotIndex,fidelity,normalized
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-dir',type=Path,required=True)
    parser.add_argument('--database',type=Path,required=True)
    parser.add_argument('--model-run',type=Path,required=True)
    parser.add_argument('--protocol',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False)
    protocol=json.loads(args.protocol.read_text())
    manifest=dict(status='running',source_sha256=source_hash(),script_sha256=file_hash(Path(__file__)),
                  protocol=protocol,faiss=faiss.__version__,torch=torch.__version__,numpy=np.__version__,
                  config=dict(m=16,ef_construction=80,ef_search=[64,128],k=10,probes=1000,seed=2026,
                              faiss_threads=1,encoder_threads=4,snapshot_ms=protocol['train_end_ms']),
                  scope='engineering fidelity on fixed available catalog, not next-click quality or final evaluation',
                  limitations=['provided embedding training/time provenance unknown',
                               'later articles omitted from fixed snapshot in both exact and approximate indexes',
                               '1000 uniformly sampled validation requests, not full query benchmark',
                               'warm single-process CPU measurements; not service/SLA or end-to-end latency'])
    started=time.perf_counter();connection=None
    write_json(args.output/'manifest.json',manifest)
    try:
        for name,key in [('train_click_log.csv','train_csv_sha256'),('articles.csv','articles_csv_sha256')]:
            if file_hash(args.data_dir/name)!=protocol[key]:raise ValueError('Input differs from protocol')
        model_info=json.loads((args.model_run/'manifest.json').read_text())
        if model_info['status']!='succeeded' or model_info['protocol']!=protocol:
            raise ValueError('Expected successful fixed-protocol encoder run')
        torch.set_num_threads(4);torch.use_deterministic_algorithms(True);faiss.omp_set_num_threads(1)
        connection=sqlite3.connect(f'file:{args.database.resolve()}?mode=ro',uri=True)
        store=VectorStore(args.data_dir,connection)
        manifest['inputs']={name:dict(bytes=(args.data_dir/name).stat().st_size,sha256=file_hash(args.data_dir/name))
                            for name in ['article_vectors.f32','article_ids.i64','VECTOR_MANIFEST.json']}
        if {key:value['sha256'] for key,value in manifest['inputs'].items()}!=model_info['input_hashes']:
            raise ValueError('Vector bytes differ from original encoder run')
        manifest['model_sha256']=file_hash(args.model_run/'model_state.pt')
        manifest['source_vector_rows']=len(store.ids)
        model=TwoTower()
        model.load_state_dict(torch.load(args.model_run/'model_state.pt',map_location='cpu',weights_only=True))
        model.eval()
        # Full cohort reconstruction, including identical tied-event semantics.
        source=args.output/'query_source';source.mkdir()
        features,records,info=build_examples(connection,store,source,protocol['validation_end_ms'],
                                             request_after=protocol['train_end_ms'])
        if len(records)!=114855:raise ValueError('Validation cohort differs from P0')
        chosen=np.sort(np.random.default_rng(2026).choice(len(records),size=1000,replace=False))
        probe_records=np.array(records[chosen][:,[0,1,2,5]],copy=True)
        selected=np.flatnonzero(store.created<=protocol['train_end_ms'])
        if len(selected)!=253708:raise ValueError('Unexpected full snapshot catalog size')
        with torch.inference_mode():
            queries=model.users(torch.from_numpy(np.array(features[chosen],copy=True))).numpy()
            catalog=np.empty((len(selected),64),dtype='float32')
            for start in range(0,len(selected),4096):
                rows=selected[start:start+4096]
                catalog[start:start+len(rows)]=model.items(torch.from_numpy(store.vectors(rows))).numpy()
        ids=np.array(store.ids[selected],copy=True);created=store.created[selected]
        np.save(args.output/'probe_records.npy',probe_records)
        np.save(args.output/'probe_queries.npy',queries)
        np.save(args.output/'catalog_ids.npy',ids)
        manifest['validation_query_source']=info
        manifest['probe_records_sha256']=file_hash(args.output/'probe_records.npy')
        manifest['catalog_ids_sha256']=file_hash(args.output/'catalog_ids.npy')
        manifest['catalog_rows']=len(ids)
        write_json(args.output/'manifest.json',manifest)
        print(f'Encoded {len(ids)} snapshot items and {len(queries)} probes; full source {len(store.ids)} rows.',flush=True)
        clock=time.perf_counter()
        exact=SnapshotIndex(ids,catalog,created,protocol['train_end_ms'])
        exact_build=time.perf_counter()-clock
        exact_scores,exact_ids,exact_time=benchmark(exact,queries,probe_records[:,2],10)
        result=dict(catalog_rows=len(ids),dimension=64,queries=len(queries),exact_build_seconds=exact_build,
                    exact_timing=exact_time,exact_vector_payload_bytes=len(ids)*64*4,hnsw={})
        np.savez_compressed(args.output/'exact_neighbors.npz',scores=exact_scores,ids=exact_ids)
        print('Exact benchmark complete; building HNSW.',flush=True)
        clock=time.perf_counter()
        approximate=SnapshotIndex(ids,catalog,created,protocol['train_end_ms'],kind='hnsw',m=16,ef_construction=80)
        result['hnsw_build_seconds']=time.perf_counter()-clock
        for width in [64,128]:
            scores,picked,timing=benchmark(approximate,queries,probe_records[:,2],10,ef_search=width)
            quality=fidelity(exact_ids,picked,exact_scores,scores)
            result['hnsw'][str(width)]=dict(fidelity=quality,timing=timing,
                                          batch_speed_ratio=exact_time['batch_median_ms']/timing['batch_median_ms'],
                                          single_p50_speed_ratio=exact_time['single_query_ms']['p50']/timing['single_query_ms']['p50'],
                                          strict_recall_gate_passed=quality['mean_strict_neighbor_recall']>=.95)
            np.savez_compressed(args.output/f'hnsw_{width}_neighbors.npz',scores=scores,ids=picked)
            print(json.dumps(dict(ef_search=width,**quality)),flush=True)
        faiss.write_index(approximate.index,str(args.output/'hnsw.faiss'))
        result['hnsw_serialized_bytes']=(args.output/'hnsw.faiss').stat().st_size
        restored=faiss.read_index(str(args.output/'hnsw.faiss'))
        restored_scores,positions=restored.search(normalized(queries),10)
        if not np.array_equal(ids[positions],picked):raise ValueError('Serialized index changed neighbor IDs')
        if not np.allclose(restored_scores,scores,atol=1e-6):raise ValueError('Serialized index changed scores')
        result['serialized_roundtrip_identical']=True
        manifest['index_sha256']=file_hash(args.output/'hnsw.faiss')
        manifest['status']='succeeded'
        write_json(args.output/'benchmark.json',result)
    except Exception as error:
        manifest['status']='failed';manifest['error']=f'{type(error).__name__}: {error}'
        raise
    finally:
        if connection:connection.close()
        manifest['wall_seconds']=time.perf_counter()-started
        write_json(args.output/'manifest.json',manifest)


if __name__=='__main__':main()
