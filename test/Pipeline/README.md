# DSPSR functional pipeline tests

## Prepare environment

```bash
# prepare python env
python3 -m venv .venv
source .venv/bin/activate

# install dependencies
pip install poetry
poetry install

# validate framework installation
poetry run python -m pipeline -h
usage: python -m pipeline [-h] --config CONFIG [-v] [--test_case_id TEST_CASE_ID] [--marker MARKER]

DSPSR pipeline test controller

options:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config (YAML) file containing DSPSR pipeline test configuration.
  -v, --verbose         Increase verbosity level (-v, -vv, -vvv).
  --test_case_id TEST_CASE_ID
                        The specific test case ID to be executed. If not provided, all test cases are executed.
  --marker MARKER       Marker filename created upon output folder creation.
```

## Run tests

```bash
# run unit tests
poetry run pytest
# run all pipeline tests
poetry run python -m pipeline --config config.yaml
```
