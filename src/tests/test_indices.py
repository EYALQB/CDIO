import numpy as np
import pytest
from src.indices import compute_ndwi

@pytest.mark.parametrize(
    "green, nir, expected", 
    [
        # Caso normal
        (np.array([[0.5]], dtype=np.float32), np.array([[0.2]], dtype=np.float32),
         np.array([[(0.5 - 0.2) / (0.5 + 0.2)]], dtype=np.float32)),

        # Denominador cero -> NaN
        (np.array([[0.0]], dtype=np.float32), np.array([[0.0]], dtype=np.float32),
         np.array([[np.nan]], dtype=np.float32)),

        # Valores iguales -> 0
        (np.array([[1.0]], dtype=np.float32), np.array([[1.0]], dtype=np.float32),
         np.array([[0.0]], dtype=np.float32)),

        # Valores faltantes -> NaN
        (np.array([[np.nan]], dtype=np.float32), np.array([[0.2]], dtype=np.float32),
         np.array([[np.nan]], dtype=np.float32)),
    ]
)
def test_compute_ndwi_parametrized(green, nir, expected):
    """Test con varios casos: normal, denom=0, iguales, NaN."""
    result = compute_ndwi(green, nir)
    if np.isnan(expected[0, 0]):
        assert np.isnan(result[0, 0])
    else:
        assert np.allclose(result, expected, atol=1e-6)


def test_determinismo_y_no_modifica_entrada():
    """Verifica que la función es determinista y no cambia las entradas."""
    g = np.array([[0.3, 0.6]], dtype=np.float32)
    n = np.array([[0.2, 0.5]], dtype=np.float32)
    g_copy = g.copy()
    n_copy = n.copy()

    r1 = compute_ndwi(g, n)
    r2 = compute_ndwi(g, n)

    assert np.allclose(r1, r2)  # mismo resultado siempre
    assert np.array_equal(g, g_copy)  # no modifica g
    assert np.array_equal(n, n_copy)  # no modifica n


def test_ndwi_array_vectorizado():
    """Verifica que funciona con arrays 2D reales."""
    g = np.array([[0.2, 0.4], [0.6, 0.8]], dtype=np.float32)
    n = np.array([[0.1, 0.3], [0.6, 0.2]], dtype=np.float32)
    result = compute_ndwi(g, n)
    expected = (g - n) / (g + n)
    assert np.allclose(result, expected, equal_nan=True)


def test_ndwi_valores_negativos():
    """Soporta valores negativos sin fallar."""
    g = np.array([[-0.5, 0.5]], dtype=np.float32)
    n = np.array([[0.2, -0.2]], dtype=np.float32)
    result = compute_ndwi(g, n)
    expected = (g - n) / (g + n)
    assert np.allclose(result, expected, equal_nan=True)


def test_ndwi_todo_ceros():
    """Cuando todo es cero -> todo NaN."""
    g = np.zeros((2, 2), dtype=np.float32)
    n = np.zeros((2, 2), dtype=np.float32)
    result = compute_ndwi(g, n)
    assert np.all(np.isnan(result))


def test_ndwi_valores_grandes():
    """Valores muy grandes no deben romper la función."""
    g = np.array([[1e10, 1e12]], dtype=np.float32)
    n = np.array([[1e10, 1e12]], dtype=np.float32)
    result = compute_ndwi(g, n)
    assert np.allclose(result, [[0.0, 0.0]], atol=1e-6)


def test_ndwi_nan_y_valores_validos():
    """Sólo las posiciones con NaN deben quedar como NaN."""
    g = np.array([[np.nan, 0.6]], dtype=np.float32)
    n = np.array([[0.3, 0.2]], dtype=np.float32)
    result = compute_ndwi(g, n)
    assert np.isnan(result[0, 0])
    assert not np.isnan(result[0, 1])


def test_ndwi_shape_coherente():
    """La salida debe tener la misma forma que la entrada."""
    g = np.random.rand(50, 60).astype(np.float32)
    n = np.random.rand(50, 60).astype(np.float32)
    result = compute_ndwi(g, n)
    assert result.shape == g.shape
