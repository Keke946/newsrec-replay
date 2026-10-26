"""Separate opt-in closed-batch replay API; existing static API unchanged."""
from .serving import ContractError
from .refresh_journal import RefreshJournalError

def create_refresh_app(engine):
    from fastapi import FastAPI,HTTPException
    from pydantic import BaseModel,ConfigDict,StrictInt,Field
    class Event(BaseModel):
        model_config=ConfigDict(extra='forbid')
        event_id:StrictInt=Field(ge=0)
        user_id:StrictInt=Field(ge=0)
        article_id:StrictInt=Field(ge=0)
    class Batch(BaseModel):
        model_config=ConfigDict(extra='forbid')
        timestamp:StrictInt=Field(ge=0)
        events:list[Event]=Field(min_length=1,max_length=10000)
        score_event_ids:list[StrictInt]|None=None
    app=FastAPI(title='NewsRec Precomputed Refresh Replay')
    @app.get('/health')
    def health():
        with engine.lock:
            if engine.closed:raise HTTPException(status_code=503,detail='Replay engine closed')
            return dict(status='ok',mode='precomputed_replay',observed_through=engine.observed_through)
    @app.post('/replay/batch')
    def batch(request:Batch):
        try:return engine.process_batch(request.timestamp,[r.model_dump() for r in request.events],request.score_event_ids)
        except RefreshJournalError as e:raise HTTPException(status_code=503,detail=str(e)) from e
        except ContractError as e:raise HTTPException(status_code=503 if engine.closed else 409,detail=str(e)) from e
    @app.post('/replay/checkpoint')
    def checkpoint():
        method=getattr(engine,'checkpoint',None)
        if method is None:raise HTTPException(status_code=409,detail='Snapshot mode not enabled')
        try:return method()
        except (RefreshJournalError,OSError) as e:raise HTTPException(status_code=503,detail=str(e)) from e
    return app
