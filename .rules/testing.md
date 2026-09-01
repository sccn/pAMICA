# Testing Standards - NO MOCKS Policy

## Core Philosophy: Test Reality, Not Fiction
**Why NO MOCKS?** Mocks test your assumptions, not your code.  
**Real bugs** hide in integration points, not unit logic.  
**Better approach:** No test is better than a false-confidence mock test.

## [STRICT] NO MOCKS, NO FAKE DATA
Never use mocks, stubs, or fake datasets. If real testing isn't possible, don't write tests.
- **No mock objects** - Use real implementations
- **No mock datasets** - Use actual sample data
- **No stub services** - Connect to real test instances
- **Alternative:** Ask user for sample data or test environment setup

## When to Write Tests
- **DO:** Test with real data and actual dependencies
- **DO:** Use test databases with real schemas
- **DO:** Test against actual file systems
- **DON'T:** Write tests if only mocks would work
- **DON'T:** Create artificial test scenarios

## Test Structure
```
tests/
  conftest.py          # Real test fixtures
  sample_data/         # Actual data samples (user-provided)
    valid/
    invalid/
  integration/         # Tests with real dependencies
    test_database.py   # Real DB connection
    test_api.py        # Real API calls
```

## Frameworks (Language-Specific)
- **Python:** `pytest` with real fixtures
- **JavaScript:** `vitest` or `jest` (no mocking libs)
- **Database:** Use test DB with real migrations
- **APIs:** Test against staging/local instances

## Writing Real Tests
```python
# GOOD: Tests actual behavior
def test_user_creation(real_db):
    """Tests that users are actually persisted."""
    user = User.create(email="test@example.com")
    # This catches: ORM issues, DB constraints, connection problems
    assert real_db.query(User).filter_by(email="test@example.com").first()

# BAD: Tests nothing meaningful
# def test_user_creation(mock_db):  # NO!
#     mock_db.return_value = User()  # Tests that Python works?
```

**Ask:** What am I actually testing? Would this catch real bugs?

## Test Data Management
- **Sample data:** Request from user or use production samples
- **Test databases:** Use Docker containers or test instances
- **File fixtures:** Use actual files, not generated ones
- **API testing:** Point to real test endpoints

## CI Integration
- Run tests with real test environment
- Skip tests if environment unavailable
- Document required test infrastructure
- See `ci_cd.md` for pipeline setup

## Sanctioned Exception: Error-Injection Subclasses
Testing an error-handling branch sometimes requires the error to fire on real
data that does not naturally produce it. The sanctioned pattern (established
in `test_ng_rank_deficient.py`, `test_mlx_sharing.py::_force_merged_column`,
and the `_RaiseForSeeds` subclasses of the restart tests) is a subclass that
raises the trigger exception at the real call site, or forces the triggering
state, while every other code path stays the real implementation on real
sample data. This is NOT the forbidden mock pattern: nothing returns a
fabricated result, and the injected exception type must be one the real call
site can actually raise. Never use a subclass to bypass or fake the numeric
path itself.

A lighter direct-call variant of the same pattern (no subclass needed): call
the real method directly on a hand-built input that forces an organically
unreachable branch, instead of driving a full fit to reach it. Established in
`test_numpy_reject.py`/`test_mlx_reject.py`'s `_reject_outliers(ll_vec)` calls
with a NaN-poisoned `ll_vec` -- the all-rejected branch is provably impossible
from a real fit (the max of any finite log-likelihood vector is always kept),
so a real fit can never exercise it. Same rule applies: the method under test
runs its real, unmodified logic; only the input is hand-built, and only to
reach a branch the method's own code proves is otherwise unreachable.

A third variant proves a code path is NOT taken: a call-counting spy wraps a
real method with `monkeypatch.setattr`, recording each call (e.g. its input
shape) into a list before delegating to the original, unmodified
implementation -- established in `test_mlx_export.py`'s
`test_write_amica_output_makes_no_extra_forward_pass`, which wraps `_forward`
this way to prove `write_amica_output` makes zero E-step passes (it reads the
LLt stash instead, issue #157). The wrapped method still runs its real logic
end to end; the spy only observes call count/arguments, never substitutes a
fabricated return value.

## When Real Testing Seems Impossible
**Think creatively before giving up:**
- Can you use Docker for a test database?
- Can you record real API responses for replay?
- Can you get anonymized production data samples?
- Can you create a minimal test environment?

**If truly impossible:**
1. Document needs in `test_requirements.md`
2. Explain to user what's needed and why
3. Ask for:
   - Sample datasets from production
   - Test environment access
   - Sandbox API credentials
4. **Be honest:** "Without real test data, I cannot verify this works"

## The Testing Mindset
- **You're not checking boxes** - you're building confidence
- **Every test should** catch at least one real bug category
- **Think:** "Will this test save someone from a 3am wake-up call?"

---
*NO MOCKS. Real tests build real confidence. When in doubt, ask for real data.*