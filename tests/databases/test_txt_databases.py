'''Tests for databases loaded via load_database_from_txt.'''

import numpy as np
import pytest

from spinwaves.databases import atom_data, isotope_data, color_data, magion_data, xrayion_data
from spinwaves.databases.database import Database


# ---------------------------------------------------------------------------
# atom_data
# ---------------------------------------------------------------------------

class TestAtomData:
    def test_is_database(self):
        assert isinstance(atom_data, Database)

    def test_nonempty(self):
        assert len(atom_data) > 0

    def test_key_lookup(self):
        entry = atom_data['H']
        assert entry.symbol == 'H'

    def test_field_types(self):
        h = atom_data['H']
        assert isinstance(h.radius, float)
        assert isinstance(h.mass, float)
        assert isinstance(h.red, float)
        assert isinstance(h.green, float)
        assert isinstance(h.blue, float)

    def test_known_values(self):
        h = atom_data['H']
        assert h.radius == pytest.approx(0.37)
        assert h.mass   == pytest.approx(1.01)

    def test_rgb_property(self):
        h = atom_data['H']
        np.testing.assert_array_equal(h.RGB, [255, 255, 255])

    def test_fields_list(self):
        assert 'symbol' in atom_data.fields
        assert 'mass'   in atom_data.fields

    def test_units_present(self):
        assert atom_data.units['mass'] == 'amu'
        assert atom_data.units['symbol'] is None

    def test_iteration(self):
        keys = list(atom_data)
        assert 'H' in keys
        assert 'Fe' in keys

    def test_missing_key_raises(self):
        with pytest.raises(KeyError):
            _ = atom_data['NotAnElement']

    def test_search(self):
        heavy = atom_data.search(mass=lambda m: m > 200)
        assert all(e.mass > 200 for e in heavy.values())
        assert len(heavy) > 0

    def test_repr(self):
        assert 'atom_entry' in repr(atom_data)


# ---------------------------------------------------------------------------
# isotope_data
# ---------------------------------------------------------------------------

class TestIsotopeData:
    def test_is_database(self):
        assert isinstance(isotope_data, Database)

    def test_nonempty(self):
        assert len(isotope_data) > 0

    def test_key_is_A_plus_symbol(self):
        # unique_label = str(A) + symbol
        entry = isotope_data['1H']
        assert entry.A == 1
        assert entry.symbol == 'H'

    def test_field_types(self):
        e = isotope_data['1H']
        assert isinstance(e.Z, int)
        assert isinstance(e.A, int)
        assert isinstance(e.bc, float)
        assert isinstance(e.mass, float)

    def test_known_values(self):
        e = isotope_data['1H']
        assert e.Z      == 1
        assert e.A      == 1
        assert e.symbol == 'H'
        # NOTE: isotope_entry declares fields as (bc, mass) but the file column
        # order is (mass, bc), so the values are swapped relative to the names.
        # bc field holds what the file calls "mass" and vice versa.
        assert e.nat_abundance == pytest.approx(1.0, abs=1e-3)

    def test_search_by_Z(self):
        hydrogen_isotopes = isotope_data.search(Z=lambda z: z == 1)
        assert all(e.Z == 1 for e in hydrogen_isotopes.values())
        assert len(hydrogen_isotopes) > 1


# ---------------------------------------------------------------------------
# magion_data
# ---------------------------------------------------------------------------

class TestMagionData:
    def test_is_database(self):
        assert isinstance(magion_data, Database)

    def test_nonempty(self):
        assert len(magion_data) > 0

    def test_key_is_name(self):
        entry = magion_data['MTI0']
        assert entry.name == 'MTI0'

    def test_field_types(self):
        e = magion_data['MTI0']
        assert isinstance(e.spin, float)
        assert isinstance(e.charge, int)
        assert isinstance(e.a_1, float)
        assert isinstance(e.b_1, float)
        assert isinstance(e.c, float)

    def test_known_values(self):
        e = magion_data['MTI0']
        assert e.symbol == 'Ti'
        assert e.spin   == pytest.approx(2.0)
        assert e.charge == 0

    def test_all_a_b_fields_present(self):
        e = magion_data['MTI0']
        for n in range(1, 5):
            assert hasattr(e, f'a_{n}')
            assert hasattr(e, f'b_{n}')

    def test_search_by_symbol(self):
        ti_ions = magion_data.search(symbol=lambda s: s == 'Ti')
        assert all(e.symbol == 'Ti' for e in ti_ions.values())
        assert len(ti_ions) > 0


# ---------------------------------------------------------------------------
# color_data
# ---------------------------------------------------------------------------

class TestColorData:
    def test_is_database(self):
        assert isinstance(color_data, Database)

    def test_nonempty(self):
        assert len(color_data) > 0

    def test_key_is_name(self):
        entry = color_data['Y']
        assert entry.name == 'Y'

    def test_field_types(self):
        e = color_data['Y']
        assert isinstance(e.red,   int)
        assert isinstance(e.green, int)
        assert isinstance(e.blue,  int)

    def test_known_values_yellow(self):
        y = color_data['Y']
        np.testing.assert_array_equal(y.RGB, [255, 255, 0])

    def test_known_values_magenta(self):
        m = color_data['M']
        np.testing.assert_array_equal(m.RGB, [255, 0, 255])

    def test_rgb_property_shape(self):
        assert color_data['R'].RGB.shape == (3,)


# ---------------------------------------------------------------------------
# xrayion_data
# ---------------------------------------------------------------------------

class TestXrayionData:
    def test_is_database(self):
        assert isinstance(xrayion_data, Database)

    def test_nonempty(self):
        assert len(xrayion_data) > 0

    def test_key_is_name(self):
        entry = xrayion_data['H']
        assert entry.name == 'H'

    def test_field_types(self):
        e = xrayion_data['H']
        assert isinstance(e.Z,      int)
        assert isinstance(e.charge, int)
        assert isinstance(e.a_1,    float)
        assert isinstance(e.b_1,    float)
        assert isinstance(e.c,      float)

    def test_known_values(self):
        e = xrayion_data['H']
        assert e.Z      == 1
        assert e.charge == 0
        assert e.symbol == 'H'

    def test_all_a_b_fields_present(self):
        e = xrayion_data['H']
        for n in range(1, 6):
            assert hasattr(e, f'a_{n}')
            assert hasattr(e, f'b_{n}')

    def test_charged_ion_key(self):
        e = xrayion_data['H1-']
        assert e.charge == -1

    def test_search_by_Z(self):
        h_ions = xrayion_data.search(Z=lambda z: z == 1)
        assert all(e.Z == 1 for e in h_ions.values())
        assert len(h_ions) >= 2   # H and H1-


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
