`BasicSelection` — the public alias for the lowering's output vocabulary,
`tuple[int | slice | None, ...]`, exported at the package root. The name is
the destination contract's: it is what `zarr.AsyncArray.getitem` calls its
selection parameter, and what NumPy calls basic indexing. Use it to annotate a
consumer-owned I/O protocol (`async def getitem(self, selection:
BasicSelection)`), as the asyncio example now does.
