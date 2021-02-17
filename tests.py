import psmatching.match as psm
import pytest


path = "./sample.csv"
model = "CASE ~ AGE + ENCODED_SEX + ENCODED_RACE + ENCODED_CCI_GROUP"
k = "5"
gap = 180

m = psm.PSMatch(path, model, k, gap)


def test_class():
    assert m.path
    assert m.model
    assert m.k
    assert m.gap


def test_prep_data():
    global m
    m.prepare_data()
    assert not m.df.empty


def test_match():
    global m
    m.match_by_neighbor(caliper = 0.005)
    assert not m.matched_controls.empty


