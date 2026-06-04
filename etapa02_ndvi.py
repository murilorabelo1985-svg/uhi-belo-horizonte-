# -*- coding: utf-8 -*-
# =============================================================================
#  TESE / ARTIGO UHI-BH  |  PIPELINE REPRODUTIVEL  v2
#  ETAPA 2  —  NDVI com máscara de nuvens + saturação, por cena
#
#  Pré-requisito: Etapa 0 e Etapa 1 concluídas sem erros.
#
#  O que este script faz:
#    1. Lê as bandas RED (SR_B4) e NIR (SR_B5) de cada cena
#    2. Aplica os fatores de escala da Coleção 2 (SR_MULT / SR_ADD)
#    3. Mascara pixels ruins: nuvem/sombra (QA_PIXEL) + saturação (QA_RADSAT)
#    4. Calcula NDVI = (NIR - RED) / (NIR + RED)
#    5. Salva um GeoTIFF de NDVI por cena em saida/T1/NDVI_cenas/
#    6. Imprime estatísticas para conferência
#
#  Como rodar: mesmo procedimento da Etapa 1 — abrir no editor do QGIS
#  e clicar no triângulo verde.
# =============================================================================

from osgeo import gdal
import numpy as np
import os, glob

gdal.UseExceptions()

# =============================================================================
#  CONFIG  —  mesmos caminhos da Etapa 1, não precisa mudar
# =============================================================================

RAIZ   = r"C:\dados_landsat"
EPOCAS = ["T3"]
SAIDA  = r"C:\dados_landsat\saida"
NODATA = -9999.0

# =============================================================================
#  PARÂMETROS FIXOS
# =============================================================================

SR_MULT = 2.75e-05
SR_ADD  = -0.2

SFX_RED = "_SR_B4.TIF"
SFX_NIR = "_SR_B5.TIF"
SFX_QAP = "_QA_PIXEL.TIF"
SFX_QAR = "_QA_RADSAT.TIF"

# Bits de saturação no QA_RADSAT para NDVI:
#   RED = banda 4 → bit 3
#   NIR = banda 5 → bit 4
RADSAT_BITS_NDVI = (3, 4)

# Intervalo válido de NDVI (-1 a 1, com folga para arredondamento)
NDVI_MIN = -1.0
NDVI_MAX =  1.0

# =============================================================================
#  FUNÇÕES AUXILIARES  (idênticas às da Etapa 1)
# =============================================================================

def achar_banda(pasta, sufixo):
    achados = [f for f in glob.glob(os.path.join(pasta, "*"))
               if f.upper().endswith(sufixo.upper())]
    if len(achados) == 0:
        raise FileNotFoundError(f"Banda '{sufixo}' não encontrada em:\n  {pasta}")
    if len(achados) > 1:
        raise RuntimeError(f"Múltiplos arquivos '{sufixo}' em {pasta}")
    return achados[0]


def ler_banda(caminho, dtype=np.float32):
    ds = gdal.Open(caminho)
    if ds is None:
        raise IOError(f"GDAL não conseguiu abrir: {caminho}")
    arr = ds.GetRasterBand(1).ReadAsArray().astype(dtype)
    gt, proj = ds.GetGeoTransform(), ds.GetProjection()
    ds = None
    return arr, gt, proj


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


def mascara_nuvem(qa_pixel):
    """Bits 0-4 do QA_PIXEL: fill, nuvem dilatada, cirrus, nuvem, sombra."""
    qa = qa_pixel.astype(np.uint16)
    ruim = np.zeros(qa.shape, dtype=bool)
    for bit in (0, 1, 2, 3, 4):
        ruim |= ((qa >> bit) & 1).astype(bool)
    return ruim


def mascara_saturacao(qa_radsat, bits):
    """Bits de saturação para as bandas RED e NIR no QA_RADSAT."""
    qa = qa_radsat.astype(np.uint16)
    sat = np.zeros(qa.shape, dtype=bool)
    for bit in bits:
        sat |= ((qa >> bit) & 1).astype(bool)
    return sat


def listar_cenas(pasta_epoca):
    return sorted([
        os.path.join(pasta_epoca, d)
        for d in os.listdir(pasta_epoca)
        if os.path.isdir(os.path.join(pasta_epoca, d))
        and d.startswith("LC0")
    ])

# =============================================================================
#  ETAPA 2  —  NDVI por cena
# =============================================================================

def etapa2_ndvi():
    print("=" * 65)
    print("ETAPA 2  —  NDVI com máscara de nuvens e saturação")
    print("=" * 65)

    for epoca in EPOCAS:
        pasta_ep  = os.path.join(RAIZ, epoca)
        pasta_out = os.path.join(SAIDA, epoca, "NDVI_cenas")
        os.makedirs(pasta_out, exist_ok=True)
        cenas = listar_cenas(pasta_ep)

        print(f"\n{'─'*65}")
        print(f"ÉPOCA {epoca}  —  {len(cenas)} cenas")
        print(f"{'─'*65}")

        for pasta in cenas:
            partes   = os.path.basename(pasta).split("_")
            data_str = partes[3] if len(partes) > 3 else "?"
            print(f"\n  {data_str}")

            # Lê as quatro bandas necessárias
            p_red = achar_banda(pasta, SFX_RED)
            p_nir = achar_banda(pasta, SFX_NIR)
            p_qap = achar_banda(pasta, SFX_QAP)
            p_qar = achar_banda(pasta, SFX_QAR)

            red_dn, gt, proj = ler_banda(p_red, np.float64)
            nir_dn, _,  _    = ler_banda(p_nir, np.float64)
            qa_pix, _,  _    = ler_banda(p_qap, np.uint16)
            qa_rad, _,  _    = ler_banda(p_qar, np.uint16)

            # 1) Converte DN → reflectância de superfície
            red = red_dn * SR_MULT + SR_ADD
            nir = nir_dn * SR_MULT + SR_ADD

            # 2) Calcula NDVI (evita divisão por zero)
            soma = nir + red
            ndvi = np.where(soma != 0, (nir - red) / soma, NODATA)

            # 3) Máscaras combinadas
            m_nuv = mascara_nuvem(qa_pix)
            m_sat = mascara_saturacao(qa_rad, RADSAT_BITS_NDVI)
            m_fill = (red_dn == 0) | (nir_dn == 0)
            m_fora = (ndvi < NDVI_MIN) | (ndvi > NDVI_MAX)  # outliers físicos
            invalido = m_nuv | m_sat | m_fill | m_fora

            ndvi[invalido] = NODATA

            # 4) Estatísticas
            validos = ndvi[ndvi != NODATA]
            n_tot = ndvi.size
            n_val = validos.size
            pct_mask = 100.0 * (n_tot - n_val) / n_tot

            # Conta pixels saturados especificamente (excluindo os já nublados)
            so_sat = m_sat & ~m_nuv & ~m_fill
            n_sat  = int(so_sat.sum())
            pct_sat = 100.0 * n_sat / n_tot

            if n_val > 0:
                print(f"    Mascarados: {pct_mask:.1f}% total "
                      f"(saturação exclusiva: {pct_sat:.2f}%)")
                print(f"    NDVI válido: min={validos.min():.3f}  "
                      f"média={validos.mean():.3f}  "
                      f"max={validos.max():.3f}")
            else:
                print(f"    [AVISO] 100% mascarado!")

            # 5) Salva
            cam_out = os.path.join(pasta_out, data_str + "_NDVI.tif")
            salvar_float32(cam_out, ndvi, gt, proj)
            print(f"    Salvo: {cam_out}")

    print(f"\nETAPA 2 concluída. Arquivos em: {os.path.join(SAIDA,'<época>','NDVI_cenas')}\n")


# =============================================================================
#  EXECUÇÃO
# =============================================================================

etapa2_ndvi()
