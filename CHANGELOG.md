# 📜 CHANGELOG – CDIO Project

## [28/09/2025] – Setmana 5
### 🔧 Correccions i millores
- *AOI CRS fix*: ara load_aoi() assegura que l’AOI estigui en EPSG:4326 (lat/lon).  
- *Query STAC millorada*: substituïda geometry per bbox → es retornen només tiles de Castelldefels (EPSG:32631).  
- **Funció clip_to_aoi()**: retalla les bandes Sentinel-2 exactament a l’AOI.  
- Afegits scripts de debug (debug_aoi.py, debug_query.py) per validar AOI i resposta STAC.  
- Eliminats després d’usar-los, per mantenir el repo net.  
- *Neteja de vores fora d’AOI*: ús de mask(..., filled=False) per obtenir màscara i marcar píxels fora d’AOI com a NaN.  
- *Export amb NODATA*: save_geotiff() actualitzat per guardar NaN com a -9999 i establir nodata=-9999 al GeoTIFF → als SIG es mostra transparent.  

---

## [Setmana 3] – Inicialització del pipeline
- Implementació de src/stac_download.py.  
- Gestió de descàrregues amb multithread i reintents.  
- Mòdul logger.py per centralitzar logs.  
- Configuració via config.yaml.  
- Visualització de bandes, composició RGB i NDWI.  
- .venv/, .tif, .jp2 i fitxers temporals afegits a .gitignore.
