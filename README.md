# CDIO – Coastal AOI Sentinel-2 Downloader & NDWI

**Aquest projecte permet:**

1.  Descarregar imatges **Sentinel-2 L2A** d’una AOI costanera definida en un fitxer **GeoJSON**.  
2.  Filtrar per dates i cobertura de núvols.  
3.  Descarregar **multithread** les bandes necessàries (Red, Green, Blue, NIR).  
4.  Visualitzar bandes, composició **RGB** i índex **NDWI**.  
5.  Configurable mitjançant **`config.yaml`**.  
6.  Logs centralitzats amb un sistema robust de detecció i gestió d’errors.  

---

## ✅ Canvis aplicats fins a la **Setmana 3**

###  Gestió de descàrrega i dades
- Implementació de `src/stac_download.py`.  
- Funció **`load_aoi()`** per llegir AOI des de `geojson`.  
- Funció **`query_stac()`** per consultar l’API STAC d’Element84.  
- Suport per **filtrar per núvols** (`max_cloud`).  
- Descàrrega **multithread** amb reintents i validació de mida mínima.  
- Carpeta `data/` buidada al repo, però mantinguda com a estructura necessària.  

###  Robustesa i logs
- Afegit mòdul `src/utils/logger.py` per gestionar logs.  
- Substitució de `print()` per `logger.info()`, `logger.warning()`, `logger.error()`.  
- Gestió d’errors en la descàrrega i lectura de TIFF (fitxers corromputs o incomplets es descarten).  

###  Configuració amb YAML
- Afegit fitxer **`config.yaml`** amb:  
  - `aoi_file`: path al fitxer GeoJSON amb AOI.  
  - `date_start` i `date_end`: interval temporal.  
  - `max_cloud`: % màxim de núvols per acceptar imatges.  
  - `n_images`: nombre d’imatges a processar.  
  - `out_dir`: carpeta de sortida.  
  - `max_workers`: fils en paral·lel per descàrrega.  
- El `main.py` ara carrega tots els paràmetres des de **`config.yaml`**.  

###  Visualització i processament
- Lectura segura de bandes (`red`, `green`, `blue`, `nir`) amb comprovació d’errors.  
- Creació de composició **RGB True Color**.  
- Càlcul i visualització de l’índex **NDWI (Green vs NIR)**.  

### 🧹 Git i neteja
- Eliminat `.venv/` del repo i afegit a `.gitignore`.  
- Afegits també `.tif`, `.jp2` i fitxers temporals per evitar arxius pesats al Git.  
- Repo net amb només el codi i la configuració necessària. 
