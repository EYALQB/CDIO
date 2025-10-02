# CDIO – Coastal AOI Sentinel-2 Downloader & NDWI

**Aquest projecte permet:**

1. Descarregar imatges **Sentinel-2 L2A** d’una AOI costanera definida en un fitxer **GeoJSON**.  
2. Filtrar per dates i cobertura de núvols.  
3. Descarregar **multithread** les bandes necessàries (Red, Green, Blue, NIR).  
4. Retallar les imatges exactament a l’**AOI**.  
5. Visualitzar bandes individuals, composició **RGB True Color** i índex **NDWI**.  
6. Configurable mitjançant **`config.yaml`**.  
7. Logs centralitzats amb un sistema robust de detecció i gestió d’errors.  

## Recordatori d’ús

### 🔹 Procés bàsic (Linux/Mac)
1. Activar l’entorn virtual:
   ```bash
   source .venv/bin/activate
   ````
2. Actualitzar el codi:
  ```bash
  git pull origin main
  ````
3. Executar el main
  ```bash
  python main.py
  ````

### 🔹 Procés bàsic (Windows)
1. Activar l’entorn virtual:
  ```bash
  .venv\Scripts\activate
  ````
2. Actualitzar el codi:
  ```bash
  git pull origin main
  ````
3. Executar el main
  ```bash
  python main.py
  ````

### 🔹 Notes
 - No pujar dades grans: la carpeta data/sentinel2/ ha d'estar buida abans de penjar-se al git.
 - Si afegeixes canvis de codi:
 ```bash
 git add -A
 git commit -m "Missatge clar del canvi"
 git push origin main
````

📖 Consulta també el [CHANGELOG](CHANGELOG.md) per veure l’històric de canvis.

