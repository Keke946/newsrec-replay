"""Optional FastAPI adapter. No training, file loading or server on import."""
from .serving import ContractError
from .journal import JournalError


def create_app(engine):
    from fastapi import FastAPI,HTTPException
    from pydantic import BaseModel,ConfigDict,Field,StrictInt

    class Request(BaseModel):
        model_config=ConfigDict(extra='forbid')
        user_id:StrictInt=Field(ge=0)
        timestamp:StrictInt=Field(ge=0)
        current_article:StrictInt|None=None
        k:StrictInt=Field(default=10,ge=1,le=10)

    class Event(BaseModel):
        model_config=ConfigDict(extra='forbid')
        event_id:StrictInt=Field(ge=0)
        user_id:StrictInt=Field(ge=0)
        article_id:StrictInt=Field(ge=0)

    class Batch(BaseModel):
        model_config=ConfigDict(extra='forbid')
        timestamp:StrictInt=Field(ge=0)
        events:list[Event]=Field(min_length=1,max_length=10000)

    app=FastAPI(title='Temporal NewsRec Replay Prototype',version='0.1')

    @app.get('/health')
    def health():
        with engine.lock:
            if getattr(engine,'closed',False):
                raise HTTPException(status_code=503,detail='Journal engine is closed')
            return dict(status='ok',model_version=engine.version,observed_through=engine.observed_through,
                        state=getattr(engine,'durability','in-memory, one worker'),replay_cache_batches=1024)

    @app.post('/recommend')
    def recommend(request:Request):
        try:return engine.recommend(**request.model_dump())
        except ContractError as error:raise HTTPException(status_code=409,detail=str(error)) from error
        except JournalError as error:raise HTTPException(status_code=503,detail=str(error)) from error

    @app.post('/events')
    def events(batch:Batch):
        try:return engine.ingest(batch.timestamp,[row.model_dump() for row in batch.events])
        except ContractError as error:raise HTTPException(status_code=409,detail=str(error)) from error
        except JournalError as error:raise HTTPException(status_code=503,detail=str(error)) from error

    return app
