"""Release acceptance suite: core audit/ranking/replay/recovery/snapshot paths."""
import os
os.environ['OPENBLAS_NUM_THREADS']='1'
import argparse,json,sys,unittest
from pathlib import Path
root=Path(__file__).resolve().parents[1];sys.path.insert(0,str(root/'src'));sys.path.insert(0,str(root/'tests'))
names=['alignment','bootstrap','catalog','contracts','transfer','trend','time_ranker','serving','journal','refresh_replay','append_refresh_journal','state_snapshot','training_backtest','refresh_backtest','temporal_targets','fusion']
p=argparse.ArgumentParser();p.add_argument('--report',type=Path);args=p.parse_args();suite=unittest.TestSuite()
for n in names:suite.addTests(unittest.defaultTestLoader.discover(str(root/'tests'),pattern='test_'+n+'.py'))
r=unittest.TextTestRunner(verbosity=2).run(suite)
report=dict(scope='v1 release acceptance, not all research branches',modules=names,tests=r.testsRun,failures=len(r.failures),errors=len(r.errors),skipped=len(r.skipped),passed=r.wasSuccessful() and not r.skipped,excluded='optional neural/ANN research and rejected dynamic SQLite backend; see historical evidence')
if args.report:
 args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report));sys.exit(0 if report['passed'] else 1)
