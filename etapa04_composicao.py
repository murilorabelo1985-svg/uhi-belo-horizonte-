# -*- coding: utf-8 -*-
# =============================================================================
#  TESE / ARTIGO UHI-BH  |  PIPELINE REPRODUTIVEL  v2
#  ETAPA 4  —  Composição mediana de LST e NDVI por época
#
#  Pré-requisito: Etapa 3 concluída (LST_BH/ e NDVI_BH/ em saida/T1/)
#
#  O que este script faz:
#    1. Lê todos os rasters recortados de uma época (LST e NDVI separado)
#    2. Reamosta todas as cenas para uma grade comum (primeira cena válida)
#    3. Empilha os arrays num cubo 3D (cenas × linhas × colunas)
#    4. Calcula a MEDIANA pixel a pixel, ignorando nodata
#    5. Salva dois rasters finais por época:
#         T1_LST_mediana.tif
#         T1_NDVI_mediana.tif
#    6. Imprime estatísticas do composto para o artigo
#
#  Por que mediana e não média?
#    A mediana é robusta a outliers (nuvens residuais, pixels extremos).
#    Um pixel nublado em 2 de 6 cenas não contamina o resultado.
# =============================================================================

from osgeo import gdal
import numpy as np
import os, glob

gdal.UseExceptions()

# =============================================================================
#  CONFIG
# =============================================================================

EPOCAS = ["T3"]
SAIDA  = r"C:\dados_landsat\saida"
NODATA = -9999.0

# Filtro de intervalo físico plausível (remove outliers residuais)
LST_MIN,  LST_MAX  =  0.0, 75.0   # °C — fora disso é artefato
NDVI_MIN, NDVI_MAX = -1.0,  1.0

# =============================================================================
#  FUNÇÕES AUXILIARES
# =============================================================================

def listar_tifs_unicos(pasta):
    """Lista TIFs sem duplicatas (resolve o bug do glob .tif/.TIF)."""
    achados = {}
    for f in glob.glob(os.path.join(pasta, "*.tif")) + \
             glob.glob(os.path.join(pasta, "*.TIF")):
        chave = os.path.basename(f).upper()
        achados[chave] = f          # sobrescreve duplicata, mantém um só
    return sorted(achados.values())


def ler_raster(caminho, dtype=np.float32):
    ds  = gdal.Open(caminho)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(dtype)
    gt, proj = ds.GetGeoTransform(), ds.GetProjection()
    ds  = None
    return arr, gt, proj


def reamostrar_para_grade(caminho_in, gt_ref, proj_ref, nx_ref, ny_ref,
                          nodata=NODATA):
    """
    Reamostrar um raster para uma grade de referência usando gdal.Warp.
    Necessário porque cenas diferentes podem ter extensões levemente distintas
    após o recorte.
    """
    tmp = "/vsimem/tmp_resample.tif"        # arquivo em memória virtual
    opts = gdal.WarpOptions(
        format="GTiff",
        outputBounds=(gt_ref[0],
                      gt_ref[3] + ny_ref * gt_ref[5],
                      gt_ref[0] + nx_ref * gt_ref[1],
                      gt_ref[3]),
        width=nx_ref, height=ny_ref,
        resampleAlg="bilinear",
        dstNodata=nodata,
        srcNodata=nodata,
    )
    gdal.Warp(tmp, caminho_in, options=opts)
    ds  = gdal.Open(tmp)
    arr = ds.GetRasterBand(1).ReadAsArray().astype(np.float32)
    ds  = None
    gdal.Unlink(tmp)
    return arr


def salvar_float32(caminho, arr, gt, proj, nodata=NODATA):
    drv = gdal.GetDriverByName("GTiff")
    ny, nx = arr.shape
    ds = drv.Create(caminho, nx, ny, 1, gdal.GDT_Float32,
                    options=["COMPRESS=DEFLATE", "TILED=YES", "BIGTIFF=IF_SAFER"])
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    b = ds.GetRasterBand(1)
    b.WriteArray(arr.astype(np.float32))
    b.SetNoDataValue(float(nodata))
    ds.FlushCache()
    ds = None


def mediana_pixel(cubo, nodata=NODATA):
    """
    Calcula a mediana pixel a pixel de um cubo (n_cenas, ny, nx),
    ignorando nodata. Pixels onde TODAS as cenas são nodata ficam como nodata.
    """
    n, ny, nx = cubo.shape
    resultado = np.full((ny, nx), nodata, dtype=np.float32)

    # Mascara o nodata
    valido = cubo != nodata                          # bool (n, ny, nx)
    n_validos = valido.sum(axis=0)                   # int  (ny, nx)

    # Pixels com pelo menos 1 cena válida
    tem_dado = n_validos > 0

    # np.nanmedian é mais lento mas mais simples; usa nan como sentinela
    cubo_nan = cubo.copy().astype(np.float64)
    cubo_nan[~valido] = np.nan

    med = np.nanmedian(cubo_nan, axis=0).astype(np.float32)
    resultado[tem_dado] = med[tem_dado]

    return resultado, n_validos


def composicao(pasta_in, nome_saida, vmin, vmax, rotulo):
    """Lê todos os TIFs de pasta_in, compõe mediana e salva nome_saida."""
    tifs = listar_tifs_unicos(pasta_in)
    if len(tifs) == 0:
        print(f"  [AVISO] Nenhum TIF encontrado em {pasta_in}")
        return None

    print(f"\n  {rotulo}: {len(tifs)} cenas")

    # Grade de referência = primeira cena
    arr0, gt_ref, proj_ref = ler_raster(tifs[0])
    ny_ref, nx_ref = arr0.shape

    # Monta o cubo
    cubo = np.full((len(tifs), ny_ref, nx_ref), NODATA, dtype=np.float32)
    for i, tif in enumerate(tifs):
        data_str = os.path.basename(tif).split("_")[0]
        if i == 0:
            arr = arr0
        else:
            arr = reamostrar_para_grade(tif, gt_ref, proj_ref,
                                        nx_ref, ny_ref)
        # Filtra intervalo físico
        arr_clean = arr.copy()
        arr_clean[(arr != NODATA) & ((arr < vmin) | (arr > vmax))] = NODATA
        cubo[i] = arr_clean
        n_val = int((arr_clean != NODATA).sum())
        print(f"    {data_str}  pixels válidos={n_val:,}")

    # Composição mediana
    print(f"\n  Calculando mediana...")
    mediana, n_obs = mediana_pixel(cubo)

    # Estatísticas do composto
    val = mediana[mediana != NODATA]
    cobertura = 100.0 * val.size / (ny_ref * nx_ref)
    print(f"  Cobertura espacial: {cobertura:.1f}% do município")
    print(f"  {rotulo} composto: min={val.min():.2f}  "
          f"média={val.mean():.2f}  max={val.max():.2f}  dp={val.std():.2f}")
    print(f"  Pixels com ≥3 cenas válidas: "
          f"{int((n_obs>=3).sum()):,} "
          f"({100.0*int((n_obs>=3).sum())/(ny_ref*nx_ref):.1f}%)")

    salvar_float32(nome_saida, mediana, gt_ref, proj_ref)
    print(f"  Salvo: {nome_saida}")
    return mediana

# =============================================================================
#  ETAPA 4
# =============================================================================

def etapa4_composicao():
    print("=" * 65)
    print("ETAPA 4  —  COMPOSIÇÃO MEDIANA POR ÉPOCA")
    print("=" * 65)

    for epoca in EPOCAS:
        print(f"\n{'─'*65}")
        print(f"ÉPOCA {epoca}")
        print(f"{'─'*65}")

        pasta_lst  = os.path.join(SAIDA, epoca, "LST_BH")
        pasta_ndvi = os.path.join(SAIDA, epoca, "NDVI_BH")
        saida_lst  = os.path.join(SAIDA, epoca, f"{epoca}_LST_mediana.tif")
        saida_ndvi = os.path.join(SAIDA, epoca, f"{epoca}_NDVI_mediana.tif")

        lst_comp  = composicao(pasta_lst,  saida_lst,
                               LST_MIN,  LST_MAX,  "LST (°C)")
        ndvi_comp = composicao(pasta_ndvi, saida_ndvi,
                               NDVI_MIN, NDVI_MAX, "NDVI")

        # Correlação LST × NDVI no composto (só pixels válidos em ambos)
        if lst_comp is not None and ndvi_comp is not None:
            mask = (lst_comp != NODATA) & (ndvi_comp != NODATA)
            lst_v  = lst_comp[mask]
            ndvi_v = ndvi_comp[mask]
            if lst_v.size > 0:
                r = np.corrcoef(lst_v, ndvi_v)[0, 1]
                print(f"\n  Correlação LST × NDVI no composto T1: r = {r:.3f}")

    print(f"\nETAPA 4 concluída.")
    print(f"Compostos finais em: {SAIDA}/<época>/")
    print(f"  <época>_LST_mediana.tif")
    print(f"  <época>_NDVI_mediana.tif\n")


# =============================================================================
#  EXECUÇÃO
# =============================================================================

etapa4_composicao()
