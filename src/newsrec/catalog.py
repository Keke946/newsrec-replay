"""Metadata-only cold candidates. No future first-click labels are consulted."""
from bisect import bisect_right
from collections import Counter, defaultdict


class FreshCatalog:
    def __init__(self, metadata, max_age_ms=6*60*60*1000):
        self.max_age_ms=max_age_ms
        self.global_items=sorted((data['created'],item) for item,data in metadata.items())
        self.categories=defaultdict(list)
        for created,item in self.global_items:
            self.categories[metadata[item]['category']].append((created,item))
        self.metadata=metadata

    def _latest(self, items, timestamp, excluded, k):
        position=bisect_right(items,(timestamp,float('inf')))
        result=[]
        for index in range(position-1,-1,-1):
            created,item=items[index]
            if created<timestamp-self.max_age_ms:break
            if item not in excluded:
                result.append(item)
                if len(result)>=k:break
        return result

    def recommend(self, history, timestamp, seen, k=10):
        if k<=0:return []
        excluded=set(item for item,_ in history)
        # A read-only overlay avoids copying the whole observed catalog per query.
        class Excluded:
            def __contains__(self,item):return item in seen or item in excluded
        blocked=Excluded()
        counts=Counter(self.metadata[item]['category'] for item,_ in history)
        categories=sorted(counts,key=lambda category:(-counts[category],category))[:2]
        result=[]
        for category in categories:
            for item in self._latest(self.categories[category],timestamp,blocked,max(1,k//2)):
                if item not in result:result.append(item)
        if len(result)<k:
            # Global fill is also metadata-only and excludes already chosen items.
            for item in self._latest(self.global_items,timestamp,blocked,k+len(result)):
                if item not in result:result.append(item)
                if len(result)>=k:break
        return result[:k]
