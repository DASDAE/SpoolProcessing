"""
Simple test to try to reproduce dascore issue #171.
"""

import pytest

import dascore as dc

from proc import std


@pytest.fixture()
def real_terra15_data_spool():
    """Get a small spool for testing."""
    return dc.spool("test_data").update()


@pytest.fixture()
def synth_data_memory_spool():
    """Create a synthetic spool for testing."""
    spool = dc.get_example_spool(
        "random_das",
        starttime="2023-06-13T15:38:00.49953408",
        d_time=1/100,
        shape=(100, 100),
        length=60,
    )
    return spool


@pytest.fixture()
def synth_data_directory_spool(synth_data_memory_spool, tmp_path_factory):
    """Create a synthetic spool for testing."""
    path = tmp_path_factory.mktemp("synth_data_dir")
    spool_path = dc.examples.spool_to_directory(synth_data_memory_spool, path)
    return dc.spool(spool_path).update()


def test_proc_real_data(real_terra15_data_spool, tmp_path_factory):
    """Ensure the proc works on real terra15 data."""
    out = tmp_path_factory.mktemp("std_out_real_data")
    std(real_terra15_data_spool, str(out))


def test_proc_synth_data_memory(synth_data_memory_spool, tmp_path_factory):
    """Ensure the proc works on in-memory spool"""
    spool = synth_data_memory_spool
    out = tmp_path_factory.mktemp("std_synth_data_memory")
    std(spool, str(out))


def test_proc_synth_data_directory(synth_data_directory_spool, tmp_path_factory):
    """Ensure the proc works on directory data."""
    spool = synth_data_directory_spool
    out = tmp_path_factory.mktemp("std_synth_data_memory")
    std(spool, str(out))
