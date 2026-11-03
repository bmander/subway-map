from subway.crosswalk import name_similarity, norm


def test_normalization():
    assert norm("GRD CNTRL-42 ST") == "GRAND CENTRAL 42 ST"
    assert norm("Jackson Hts-Roosevelt Av") == "JACKSON HTS ROOSEVELT AV"


def test_numbers_must_match():
    assert name_similarity("59 ST", "86 St") == 0.0
    assert name_similarity("59 ST", "59 St") > 0.9


def test_prefers_tight_name_over_superset():
    # "23 ST" (F/M) once matched "Court Sq-23 St" on a tie.
    assert name_similarity("23 ST", "23 St") > name_similarity("23 ST", "Court Sq-23 St")
    assert name_similarity("50 ST", "50 St") > name_similarity("50 ST", "Bay 50 St")


def test_truncated_names_match():
    assert name_similarity("AQUEDUCT RACETR", "Aqueduct Racetrack") > 0.85
    assert name_similarity("JKSN HT-ROOSVLT", "Jackson Hts-Roosevelt Av") > 0.7
