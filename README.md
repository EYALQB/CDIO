# CDIO – Coastal AOI Sentinel-2 Downloader & NDWI

**Aquest projecte permet:**

1. Descarregar imatges **Sentinel-2 L2A** d’una AOI costanera definida en un fitxer **GeoJSON**.  
2. Filtrar per dates i cobertura de núvols.  
3. Descarregar **multithread** les bandes necessàries (Red, Green, Blue, NIR).  
4. Retallar les imatges exactament a l’**AOI**.  
5. Visualitzar bandes individuals, composició **RGB True Color** i índex **NDWI**.  
6. Configurable mitjançant **`config.yaml`**.  
7. Logs centralitzats amb un sistema robust de detecció i gestió d’errors.  

---

## ⚙️ Estructura del projecte
│── data/ # Carpeta per guardar imatges descarregades i AOIs
│ └── aoi_castelldefels.geojson
│
│── src/
│ ├── indices.py # Càlcul NDWI
│ ├── visualize.py # Funcions de visualització
│ ├── stac_download.py # Query i descàrrega d’imatges Sentinel-2
│ └── utils/
│ └── logger.py # Sistema de logs
│
│── config.yaml # Configuració general
│── main.py # Script principal
│── CHANGELOG.md # Històric de canvis setmanals


---
