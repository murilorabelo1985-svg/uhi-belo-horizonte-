# -*- coding: utf-8 -*-
# =============================================================================
#  TESE / ARTIGO UHI-BH  |  PIPELINE REPRODUTIVEL  v2
#  ETAPA 3  —  Recorte pelo limite municipal de Belo Horizonte
#
#  Pré-requisito: Etapas 1 e 2 concluídas (LST e NDVI por cena em saida/T1/)
#
#  O que este script faz:
#    1. Reprojetar o shapefile do limite municipal (SIRGAS 2000 / EPSG:31983)
#       para o CRS das imagens (WGS84/UTM 23, EPSG:32623) — na memória,
#       sem criar arquivo intermediário
#    2. Recorta cada raster LST e NDVI pelo polígono reprojetado
#    3. Salva os rasters recortados em saida/T1/LST_BH/ e saida/T1/NDVI_BH/
#    4. Imprime estatísticas dentro do município para conferência
# =============================================================================

from osgeo import gdal, ogr, osr
import numpy as np
import os, glob, tempfile

gdal.UseExceptions()
ogr.UseExceptions()
osr.UseExceptions()

# =============================================================================
#  CONFIG
# =============================================================================

RAIZ      = r"C:\dados_landsat"
EPOCAS    = ["T3"]
SAIDA     = r"C:\dados_landsat\saida"
SHAPEFILE = r"C:\dados_landsat\Vetores\LIMITE_MUNICIPIO.shp"
NODATA    = -9999.0

# =============================================================================
#  FUNÇÕES AUXILIARES
# =============================================================================

def listar_tifs(pasta):
    """Lista todos os .tif/.TIF numa pasta, ordenados."""
    return sorted(glob.glob(os.path.join(pasta, "*.tif")) +
                  glob.glob(os.path.join(pasta, "*.TIF")))


def reprojetar_shp_memoria(caminho_shp, epsg_destino):
    """
    Reprojetar um shapefile para um EPSG destino e devolver como
    arquivo temporário .shp (necessário para gdal.Warp cutline).
    Retorna o caminho do arquivo temporário.
    """
    drv = ogr.GetDriverByName("ESRI Shapefile")
    src_ds = drv.Open(caminho_shp, 0)
    src_lyr = src_ds.GetLayer()
    src_srs = src_lyr.GetSpatialRef()

    dst_srs = osr.SpatialReference()
    dst_srs.ImportFromEPSG(epsg_destino)
    dst_srs.SetAxisMappingStrategy(osr.OAMS_TRADITIONAL_GIS_ORDER)

    transform = osr.CoordinateTransformation(src_srs, dst_srs)

    # Arquivo temporário para o shapefile reprojetado
    tmp_dir  = tempfile.mkdtemp()
    tmp_path = os.path.join(tmp_dir, "limite_reproj.shp")

    drv_mem = ogr.GetDriverByName("ESRI Shapefile")
    dst_ds  = drv_mem.CreateDataSource(tmp_path)
    dst_lyr = dst_ds.CreateLayer("limite", srs=dst_srs,
                                 geom_type=src_lyr.GetGeomType())

    # Copia definição de campos
    src_defn = src_lyr.GetLayerDefn()
    for i in range(src_defn.GetFieldCount()):
        dst_lyr.CreateField(src_defn.GetFieldDefn(i))

    # Copia e reprojetar feições
    for feat in src_lyr:
        geom = feat.GetGeometryRef().Clone()
        geom.Transform(transform)
        nova = ogr.Feature(dst_lyr.GetLayerDefn())
        nova.SetGeometry(geom)
        for i in range(src_defn.GetFieldCount()):
            nova.SetField(src_defn.GetFieldDefn(i).GetNameRef(),
                          feat.GetField(i))
        dst_lyr.CreateFeature(nova)

    dst_ds.FlushCache()
    dst_ds = src_ds = None
    return tmp_path


def recortar_raster(caminho_in, caminho_out, shp_cutline, nodata=NODATA):
    """
    Recorta um raster pelo polígono do shapefile usando gdal.Warp.
    Mantém o CRS e resolução originais.
    """
    opts = gdal.WarpOptions(
        format="GTiff",
        cutlineDSName=shp_cutline,
        cropToCutline=True,
        dstNodata=nodata,
        creationOptions=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"],
    )
    resultado = gdal.Warp(caminho_out, caminho_in, options=opts)
    if resultado is None:
        raise RuntimeError(f"gdal.Warp falhou para: {caminho_in}")
    resultado = None


def estatisticas(caminho_tif, nodata=NODATA):
    """Lê um raster e devolve min/média/max/dp dos pixels válidos."""
    ds  = gdal.Open(caminho_tif)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds  = None
    val = arr[arr != nodata]
    if val.size == 0:
        return None
    return val.min(), val.mean(), val.max(), val.std(), val.size


def epsg_do_raster(caminho_tif):
    """Lê o EPSG do CRS de um raster."""
    ds  = gdal.Open(caminho_tif)
    srs = osr.SpatialReference(wkt=ds.GetProjection())
    ds  = None
    codigo = srs.GetAttrValue("AUTHORITY", 1)
    return int(codigo) if codigo else None

# =============================================================================
#  ETAPA 3  —  RECORTE
# =============================================================================

def etapa3_recorte():
    print("=" * 65)
    print("ETAPA 3  —  Recorte pelo limite municipal de BH")
    print("=" * 65)
    print(f"\nShapefile: {SHAPEFILE}")

    # Cache do shapefile reprojetado por EPSG (para não reprojetar 12 vezes)
    shp_cache = {}

    for epoca in EPOCAS:
        print(f"\n{'─'*65}")
        print(f"ÉPOCA {epoca}")
        print(f"{'─'*65}")

        for tipo, subpasta_in, subpasta_out in [
            ("LST", "LST_cenas", "LST_BH"),
            ("NDVI", "NDVI_cenas", "NDVI_BH"),
        ]:
            pasta_in  = os.path.join(SAIDA, epoca, subpasta_in)
            pasta_out = os.path.join(SAIDA, epoca, subpasta_out)
            os.makedirs(pasta_out, exist_ok=True)

            tifs = listar_tifs(pasta_in)
            print(f"\n  {tipo}: {len(tifs)} cenas → {subpasta_out}/")

            for caminho_in in tifs:
                nome     = os.path.basename(caminho_in)
                data_str = nome.split("_")[0]  # ex: 20140116

                # Descobre o EPSG da imagem e obtém shapefile reprojetado
                epsg = epsg_do_raster(caminho_in)
                if epsg not in shp_cache:
                    print(f"    Reprojetando shapefile para EPSG:{epsg}...")
                    shp_cache[epsg] = reprojetar_shp_memoria(SHAPEFILE, epsg)
                shp_reproj = shp_cache[epsg]

                caminho_out = os.path.join(pasta_out,
                                           data_str + f"_{tipo}_BH.tif")
                recortar_raster(caminho_in, caminho_out, shp_reproj)

                # Estatísticas do recorte
                stats = estatisticas(caminho_out)
                if stats:
                    vmin, vmean, vmax, vstd, nval = stats
                    print(f"    {data_str}  pixels válidos={nval:,}  "
                          f"min={vmin:.2f}  média={vmean:.2f}  "
                          f"max={vmax:.2f}  dp={vstd:.2f}")
                else:
                    print(f"    {data_str}  [AVISO] sem pixels válidos!")

    print(f"\nETAPA 3 concluída.")
    print(f"Recortes em:")
    for epoca in EPOCAS:
        print(f"  {os.path.join(SAIDA, epoca, 'LST_BH')}")
        print(f"  {os.path.join(SAIDA, epoca, 'NDVI_BH')}")
    print()


# =============================================================================
#  EXECUÇÃO
# =============================================================================

etapa3_recorte()
