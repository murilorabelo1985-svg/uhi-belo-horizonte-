# -*- coding: utf-8 -*-
# =============================================================================
#  TESE / ARTIGO UHI-BH  |  PIPELINE REPRODUTIVEL  v2
#  Série L8/L9 OLI-TIRS  |  Composição mediana por época
#
#  ETAPA 0  —  Sanidade: verifica estrutura de pastas e bandas
#  ETAPA 1  —  LST em °C com máscara de nuvens, por cena
#
#  Como rodar no QGIS:
#    Complementos → Console Python → Mostrar Editor → abrir este arquivo
#    Ajuste o bloco CONFIG abaixo e clique em Executar (triângulo verde)
#    Rode primeiro só a Etapa 0; depois descomente a chamada da Etapa 1.
#
#  Dependências: osgeo.gdal + numpy (já vêm no Python do QGIS)
# =============================================================================

from osgeo import gdal
import numpy as np
import os, glob

gdal.UseExceptions()

# =============================================================================
#  CONFIG  —  EDITE APENAS AQUI
# =============================================================================

# Pasta raiz onde estão as subpastas T1, T2, T3
RAIZ = r"C:\dados_landsat"

# Épocas disponíveis (remova as que ainda não baixou)
EPOCAS = ["T3"]          # quando T2 e T3 chegarem, mude para ["T1","T2","T3"]

# Pasta de saída (será criada automaticamente)
SAIDA = r"C:\dados_landsat\saida"

# Nodata de saída
NODATA = -9999.0

# =============================================================================
#  PARÂMETROS FIXOS  —  Coleção 2 L2, L8/L9 OLI-TIRS
# =============================================================================

# Fatores de escala — temperatura de superfície (idênticos L8 e L9)
ST_MULT = 0.00341802
ST_ADD  = 149.0          # resultado em Kelvin; subtraímos 273.15 para Celsius

# Fatores de escala — reflectância de superfície (idênticos L8 e L9)
SR_MULT = 2.75e-05
SR_ADD  = -0.2

# Sufixos das bandas (L8/L9 OLI-TIRS — todas as cenas da Rodada 1)
SFX_ST   = "_ST_B10.TIF"    # temperatura de superfície
SFX_RED  = "_SR_B4.TIF"     # vermelho (RED)  — banda 4 no OLI
SFX_NIR  = "_SR_B5.TIF"     # infravermelho próximo (NIR) — banda 5 no OLI
SFX_QAP  = "_QA_PIXEL.TIF"  # máscara de nuvens
SFX_QAR  = "_QA_RADSAT.TIF" # máscara de saturação (NIR=bit4, RED=bit3)

# Bits de saturação no QA_RADSAT para RED (B4→bit3) e NIR (B5→bit4)
RADSAT_BITS_NDVI = (3, 4)

# =============================================================================
#  FUNÇÕES AUXILIARES
# =============================================================================

def achar_banda(pasta, sufixo):
    """Encontra o único arquivo com dado sufixo (case-insensitive) na pasta."""
    padrao = os.path.join(pasta, "*")
    achados = [f for f in glob.glob(padrao)
               if f.upper().endswith(sufixo.upper())]
    if len(achados) == 0:
        raise FileNotFoundError(
            f"Banda '{sufixo}' não encontrada em:\n  {pasta}")
    if len(achados) > 1:
        raise RuntimeError(
            f"Múltiplos arquivos '{sufixo}' em {pasta}:\n  " +
            "\n  ".join(achados))
    return achados[0]


def ler_banda(caminho, dtype=np.float32):
    """Lê a primeira banda de um GeoTIFF e devolve (array, gt, proj)."""
    ds = gdal.Open(caminho)
    if ds is None:
        raise IOError(f"GDAL não conseguiu abrir: {caminho}")
    arr = ds.GetRasterBand(1).ReadAsArray().astype(dtype)
    gt   = ds.GetGeoTransform()
    proj = ds.GetProjection()
    ds   = None
    return arr, gt, proj


def salvar_float32(caminho, arr, gt, proj, nodata=NODATA):
    """Salva array float32 como GeoTIFF comprimido."""
    drv = gdal.GetDriverByName("GTiff")
    ny, nx = arr.shape
    ds = drv.Create(caminho, nx, ny, 1, gdal.GDT_Float32,
                    options=["COMPRESS=DEFLATE", "TILED=YES",
                             "BIGTIFF=IF_SAFER"])
    ds.SetGeoTransform(gt)
    ds.SetProjection(proj)
    b = ds.GetRasterBand(1)
    b.WriteArray(arr.astype(np.float32))
    b.SetNoDataValue(float(nodata))
    ds.FlushCache()
    ds = None


def mascara_ruim(qa_pixel):
    """
    Retorna máscara booleana True = pixel ruim (fill, nuvem, sombra, cirrus).
    Bits Collection 2 QA_PIXEL:
      0 = fill | 1 = nuvem dilatada | 2 = cirrus | 3 = nuvem | 4 = sombra
    """
    qa = qa_pixel.astype(np.uint16)
    ruim = np.zeros(qa.shape, dtype=bool)
    for bit in (0, 1, 2, 3, 4):
        ruim |= ((qa >> bit) & 1).astype(bool)
    return ruim


def mascara_saturacao(qa_radsat, bits):
    """Retorna True onde qualquer dos bits de saturação está ativo."""
    qa = qa_radsat.astype(np.uint16)
    sat = np.zeros(qa.shape, dtype=bool)
    for bit in bits:
        sat |= ((qa >> bit) & 1).astype(bool)
    return sat


def listar_cenas(pasta_epoca):
    """Devolve lista de subpastas de cenas dentro de uma pasta de época."""
    subs = sorted([
        os.path.join(pasta_epoca, d)
        for d in os.listdir(pasta_epoca)
        if os.path.isdir(os.path.join(pasta_epoca, d))
        and d.startswith("LC0")   # LC08 ou LC09
    ])
    return subs

# =============================================================================
#  ETAPA 0  —  SANIDADE
# =============================================================================

def etapa0_sanidade():
    print("=" * 65)
    print("ETAPA 0  —  VERIFICAÇÃO DE SANIDADE")
    print("=" * 65)
    os.makedirs(SAIDA, exist_ok=True)
    tudo_ok = True

    for epoca in EPOCAS:
        pasta_ep = os.path.join(RAIZ, epoca)
        if not os.path.isdir(pasta_ep):
            print(f"\n[ERRO] Pasta não encontrada: {pasta_ep}")
            tudo_ok = False
            continue

        cenas = listar_cenas(pasta_ep)
        print(f"\n{'─'*65}")
        print(f"ÉPOCA {epoca}  —  {len(cenas)} subpastas encontradas")
        print(f"{'─'*65}")

        if len(cenas) == 0:
            print("  [AVISO] Nenhuma subpasta LC0x encontrada nesta época.")
            tudo_ok = False
            continue

        # Usamos a primeira cena como grade de referência para a época
        gt_ref = proj_ref = nx_ref = ny_ref = None

        for pasta in cenas:
            nome = os.path.basename(pasta)
            print(f"\n  Cena: {nome}")
            ok_cena = True
            try:
                p_st  = achar_banda(pasta, SFX_ST)
                p_red = achar_banda(pasta, SFX_RED)
                p_nir = achar_banda(pasta, SFX_NIR)
                p_qap = achar_banda(pasta, SFX_QAP)
                p_qar = achar_banda(pasta, SFX_QAR)
            except FileNotFoundError as e:
                print(f"    [ERRO] {e}")
                tudo_ok = ok_cena = False
                continue

            # Abre banda ST como referência
            ds = gdal.Open(p_st)
            gt   = ds.GetGeoTransform()
            proj = ds.GetProjection()
            nx, ny = ds.RasterXSize, ds.RasterYSize
            pixel_m = gt[1]
            ds = None

            # Informa CRS na primeira cena da época
            if gt_ref is None:
                gt_ref, proj_ref, nx_ref, ny_ref = gt, proj, nx, ny
                srs = gdal.osr.SpatialReference(wkt=proj)
                epsg = srs.GetAttrValue("AUTHORITY", 1)
                nome_proj = (srs.GetAttrValue("PROJCS") or
                             srs.GetAttrValue("GEOGCS") or "?")
                print(f"    Grade de referência da época: {nx}×{ny} px | "
                      f"pixel={pixel_m:.0f}m | EPSG:{epsg}")
                print(f"    Proj: {nome_proj}")

            # Checa alinhamento entre bandas da mesma cena
            desalinhadas = []
            for p, sfx in [(p_red,SFX_RED),(p_nir,SFX_NIR),
                           (p_qap,SFX_QAP),(p_qar,SFX_QAR)]:
                d = gdal.Open(p)
                if (d.RasterXSize, d.RasterYSize) != (nx, ny) or \
                   d.GetGeoTransform() != gt:
                    desalinhadas.append(sfx)
                d = None
            if desalinhadas:
                print(f"    [AVISO] Bandas desalinhadas: {desalinhadas}")
                tudo_ok = ok_cena = False
            else:
                print(f"    Todas as bandas alinhadas. OK.")

    print("\n" + "=" * 65)
    if tudo_ok:
        print("ETAPA 0 concluída sem erros. Pode rodar a Etapa 1.")
    else:
        print("ETAPA 0 encontrou problemas. Revise os avisos antes de continuar.")
    print("=" * 65 + "\n")
    return tudo_ok


# =============================================================================
#  ETAPA 1  —  LST em °C com máscara de nuvens, por cena
# =============================================================================

def etapa1_lst():
    print("=" * 65)
    print("ETAPA 1  —  LST em °C com máscara de nuvens")
    print("=" * 65)

    for epoca in EPOCAS:
        pasta_ep = os.path.join(RAIZ, epoca)
        pasta_lst = os.path.join(SAIDA, epoca, "LST_cenas")
        os.makedirs(pasta_lst, exist_ok=True)
        cenas = listar_cenas(pasta_ep)

        print(f"\n{'─'*65}")
        print(f"ÉPOCA {epoca}  —  processando {len(cenas)} cenas")
        print(f"{'─'*65}")

        for pasta in cenas:
            nome = os.path.basename(pasta)
            # Extrai data do nome (posição fixa no Product ID)
            partes = nome.split("_")
            data_str = partes[3] if len(partes) > 3 else "?"
            print(f"\n  {data_str}  ({nome[:30]}...)")

            p_st  = achar_banda(pasta, SFX_ST)
            p_qap = achar_banda(pasta, SFX_QAP)

            st_dn,  gt, proj = ler_banda(p_st,  np.float64)
            qa_pix, _,  _    = ler_banda(p_qap, np.uint16)

            # 1) Converte DN → Kelvin → Celsius
            lst_c = st_dn * ST_MULT + ST_ADD - 273.15

            # 2) Aplica máscaras
            m_fill  = (st_dn == 0)           # pixels de preenchimento
            m_nuv   = mascara_ruim(qa_pix)   # nuvem + sombra + cirrus
            invalido = m_fill | m_nuv
            lst_c[invalido] = NODATA

            # 3) Estatísticas dos pixels válidos
            validos = lst_c[lst_c != NODATA]
            n_tot = lst_c.size
            n_val = validos.size
            pct_mask = 100.0 * (n_tot - n_val) / n_tot
            if n_val > 0:
                print(f"    Mascarados: {pct_mask:.1f}% | "
                      f"LST válida: min={validos.min():.1f} "
                      f"média={validos.mean():.1f} "
                      f"max={validos.max():.1f} °C")
            else:
                print(f"    [AVISO] 100% mascarado — cena totalmente nublada!")

            # 4) Salva
            nome_out = data_str + "_LST_C.tif"
            cam_out  = os.path.join(pasta_lst, nome_out)
            salvar_float32(cam_out, lst_c, gt, proj)
            print(f"    Salvo: {cam_out}")

    print(f"\nETAPA 1 concluída. Arquivos em: {os.path.join(SAIDA,'<época>','LST_cenas')}\n")


# =============================================================================
#  EXECUÇÃO
# =============================================================================

etapa0_sanidade()
etapa1_lst()
