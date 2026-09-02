# ecostress/

Librería espectral ECOSTRESS (ECOSTRESS Spectral Library, versión 1.0) organizada por categoría de material. Cada archivo contiene la firma espectral (reflectancia o transmitancia) de un objeto/material medido en laboratorio.

## Estructura

```
ecostress/
├── vegetation/                     # Plantas y especies (hojas, árboles, arbustos, pastos)
├── nonphotosyntheticvegetation/    # Vegetación no fotosintética (corteza, madera muerta, etc.)
├── mineral/                        # Minerales (silicatos, carbonatos, óxidos, etc.)
├── rock/                           # Rocas (ígneas, metamórficas, sedimentarias)
├── soil/                           # Suelos (alfisoles, entisoles, etc.)
├── manmade/                        # Materiales hechos por el hombre (concreto, pintura, metal)
├── water/                          # Agua, hielo, nieve, escarcha
└── meteorites/                     # Meteoritos
```

## Formato de archivo

Cada archivo `.txt` tiene un encabezado con metadatos y luego los datos espectrales en dos columnas separadas por tabulador:

```
Name: Quercus douglasii 2
Type: vegetation
Class: Tree
...
Measurement: Bidirectional reflectance
X Units: Wavelength (micrometers)
Y Units: Reflectance (percentage)

 0.3500	 6.1800
 0.3510	 6.0510
 ...
```

- **Columna X**: longitud de onda en micrómetros (µm)
- **Columna Y**: reflectancia en porcentaje (0-100) — o transmitancia para algunos espectros TIR
- El rango espectral depende de la medición: **VSWIR** (0.35–2.5 µm) o **TIR** (2.5–25 µm)
- Los archivos `*_1.txt` / espectros con `_1` en el nombre son mediciones; los `.ancillary.txt` contienen metadatos adicionales

## Uso en el notebook

```python
import glob
import numpy as np

# Cargar todos los espectros de vegetación
for file in glob.glob('ecostress/vegetation/*spectrum.txt'):
    data = np.loadtxt(file, skiprows=20)  # saltar el encabezado
    wavelengths_um = data[:, 0]
    reflectance = data[:, 1]
```

Fuente original: https://speclib.jpl.nasa.gov/ (ECOSTRESS Spectral Library, JPL)
