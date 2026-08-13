`as_basic_selection` refusals now raise `NoBasicSelectionError`, a dedicated
`ValueError` subclass exported at the package root. The documented consumer
fallback (`except NoBasicSelectionError: part.view.result()`) can no longer
silently absorb a genuine defect in the lowering, which bare `except
ValueError` did. Existing catch sites keep working: the subclass is caught by
`except ValueError` unchanged.
