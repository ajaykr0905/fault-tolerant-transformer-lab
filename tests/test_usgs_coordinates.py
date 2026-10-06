import json

import pytest

from fttl.usgs import USGSCaptureLedger, USGSValidationError, parse_feature_collection


def _payload(coordinates):
    return json.dumps(
        {
            "type": "FeatureCollection",
            "features": [
                {
                    "type": "Feature",
                    "id": "independent-coordinates-event",
                    "properties": {"updated": 42},
                    "geometry": {"type": "Point", "coordinates": coordinates},
                }
            ],
        }
    ).encode()


def _ledger_history(ledger):
    return tuple(
        tuple(ledger.connection.execute(f"SELECT * FROM {table} ORDER BY rowid").fetchall())
        for table in ("polls", "event_versions")
    )


@pytest.mark.parametrize("position", [0, 1, 2])
@pytest.mark.parametrize("sign", [-1, 1])
def test_huge_integer_coordinate_has_stable_parse_and_capture_rejection(tmp_path, position, sign):
    coordinates = [1, 2, 3]
    coordinates[position] = sign * 10**500
    malformed = _payload(coordinates)
    with pytest.raises(USGSValidationError, match="invalid coordinates"):
        parse_feature_collection(malformed, source="usgs-earthquakes-all-day")
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        ledger.capture(_payload([1, 2, 3]), feed="all-day", captured_at="2026-10-06T00:00:00Z")
        before = _ledger_history(ledger)
        with pytest.raises(USGSValidationError, match="invalid coordinates"):
            ledger.capture(malformed, feed="all-day", captured_at="2026-10-06T00:01:00Z")
        assert _ledger_history(ledger) == before
        assert len(before[0]) == len(before[1]) == 1


@pytest.mark.parametrize(
    "coordinates",
    [
        [-180.0, -90.0, -5.0],
        [-181, 91, -500],
        [0, 0],
        [1, 2, 3, 4],
        [1.0, 2.5, 3.0],
        pytest.param([10**307, -(10**307), 10**307], id="large-representable-integers"),
    ],
)
def test_existing_finite_coordinate_values_remain_accepted(tmp_path, coordinates):
    payload = _payload(coordinates)
    (version,) = parse_feature_collection(payload, source="usgs-earthquakes-all-day")
    assert version.payload["geometry"]["coordinates"] == coordinates
    with USGSCaptureLedger(tmp_path / "capture.sqlite3") as ledger:
        captured = ledger.capture(payload, feed="all-day", captured_at="2026-10-06T00:00:00Z")
        assert captured.received_events == captured.inserted_versions == 1


@pytest.mark.parametrize(
    "coordinates",
    [[True, 2], [1, False], [None, 2], [1, "2"], [1, {}], [], [1], None],
)
def test_existing_nonnumeric_coordinate_rejection_remains_stable(coordinates):
    with pytest.raises(USGSValidationError, match="invalid coordinates"):
        parse_feature_collection(_payload(coordinates), source="usgs-earthquakes-all-day")


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf")])
def test_existing_nonfinite_json_rejection_remains_stable(value):
    with pytest.raises(USGSValidationError, match="non-finite JSON number"):
        parse_feature_collection(_payload([1, value, 3]), source="usgs-earthquakes-all-day")
