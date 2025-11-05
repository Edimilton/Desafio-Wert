import re, unicodedata
import pandas as pd
import numpy as np
import streamlit as st
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# config
st.set_page_config(page_title="WERT • Vencimentos de COEs", layout="wide")
COR_FUNDO, COR_GRADE, COR_TEXTO = "#0f1116", "#2a2d36", "#eaecef"
COR_BARRA, COR_LINHA = "#34a3ff", "#ff4d4f"

# Logo
st.image("wert_logo_btg.png", width=360) 
st.title("Vencimentos de COEs")

# upload
st.sidebar.subheader("Carregar base (.xlsx)")
uploaded = st.sidebar.file_uploader("Escolha a planilha", type=["xlsx"])
if not uploaded:
    st.info("Envie a planilha .xlsx para iniciar.")
    st.stop()

# conversores
def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", str(s)).encode("ascii", "ignore").decode("utf-8")
    return re.sub(r"[^a-z0-9]+", " ", s.lower()).strip()

def to_float_brl(x):
    """Converte 'R$ 1.234,56' -> 1234.56 (NaN se inválido)."""
    if pd.isna(x): return pd.NA
    s = re.sub(r"[^0-9,.\-]", "", str(x))
    if "," in s and s.count(",") == 1:
        s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except Exception:
        return pd.NA

def brl(v):
    v = 0.0 if pd.isna(v) else float(v)
    s = f"{v:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    return f"R$ {s}"

def kpi_box(titulo: str, valor: str):
    st.markdown(
        f"<div style='padding:.9rem 1rem;border-radius:14px;background:{COR_FUNDO};"
        f"border:1px solid {COR_GRADE};'>"
        f"<div style='color:#9aa0a6;font-size:.85rem;'>{titulo}</div>"
        f"<div style='font-size:1.6rem;font-weight:700;color:{COR_TEXTO};'>{valor}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# carregamento e normalizacao
df = pd.read_excel(uploaded, engine="openpyxl")

COLMAP = {
    "Nome_Cliente":     ["nome_cliente", "nome do cliente", "cliente", "nome"],
    "Numero_Conta":     ["numero_conta", "número da conta", "conta", "n conta", "nº da conta"],
    "Valor_Aplicado":   ["valor_aplicado", "valor aplicado", "aplicado"],
    "Valor_Atual":      ["valor_atual", "valor atual", "atual"],
    "Data_Aplicacao":   ["data_aplicacao", "data aplicacao", "data de aplicacao", "dt aplicacao"],
    "Data_Vencimento":  ["data_vencimento", "data vencimento", "dt vencimento", "vencimento"],
}

# renomeio
norm_cols = {c: normalize(c) for c in df.columns}
rename = {}
for std, aliases in COLMAP.items():
    for raw, low in norm_cols.items():
        if any(normalize(a) in low for a in aliases):
            rename[raw] = std; break
df = df.rename(columns=rename)

# datas
df["Data_Aplicacao"]  = pd.to_datetime(df.get("Data_Aplicacao"),  errors="coerce").dt.date
df["Data_Vencimento"] = pd.to_datetime(df.get("Data_Vencimento"), errors="coerce").dt.date

# valores
for c in ["Valor_Aplicado", "Valor_Atual"]:
    if c in df.columns:
        df[c] = df[c].apply(to_float_brl)

# contagem de cliente 
if "Nome_Cliente" in df.columns and df["Nome_Cliente"].notna().any():
    df["ID_Cliente"] = df["Nome_Cliente"].astype(str).str.strip().str.upper()
elif "Numero_Conta" in df.columns:
    df["ID_Cliente"] = df["Numero_Conta"].astype(str).str.strip()
else:
    df["ID_Cliente"] = df.index.astype(str)

# rotulo string 
df["Data_Label"] = pd.to_datetime(df["Data_Vencimento"]).dt.strftime("%d/%m/%Y")

# tenor 
tenor = (
    pd.to_datetime(df["Data_Vencimento"], errors="coerce")
    - pd.to_datetime(df["Data_Aplicacao"], errors="coerce")
)
df["Tenor_dias"] = tenor.dt.days

# agregacao
agg = (
    df.dropna(subset=["Data_Vencimento"])
      .groupby(["Data_Vencimento", "Data_Label"], as_index=False)
      .agg(
          Valor_do_Dia=("Valor_Atual", "sum"),
          Clientes_do_Dia=("ID_Cliente", "nunique")
      )
      .sort_values("Data_Vencimento")
)
label2date = dict(zip(agg["Data_Label"], agg["Data_Vencimento"]))

# grafico
fig = make_subplots(specs=[[{"secondary_y": True}]])
# barras 
fig.add_trace(
    go.Bar(
        x=agg["Data_Label"].astype(str),
        y=pd.to_numeric(agg["Valor_do_Dia"], errors="coerce"),
        name="Valor Total a Vencer (R$)",
        orientation="v",
        marker_color=COR_BARRA,
        marker_line_color="rgba(255,255,255,0.25)",
        marker_line_width=0.6,
        hovertemplate="<b>%{x}</b><br>Valor: R$ %{y:,.2f}<extra></extra>",
    ),
    secondary_y=False
)
# linha 
fig.add_trace(
    go.Scatter(
        x=agg["Data_Label"].astype(str),
        y=pd.to_numeric(agg["Clientes_do_Dia"], errors="coerce"),
        mode="lines+markers",
        line=dict(width=2.2, color=COR_LINHA),
        marker=dict(size=6, line=dict(width=0.8, color=COR_FUNDO)),
        name="Quantidade de Clientes com Vencimento",
        hovertemplate="<b>%{x}</b><br>Clientes: %{y:.0f}<extra></extra>",
    ),
    secondary_y=True
)

fig.update_layout(
    paper_bgcolor=COR_FUNDO, plot_bgcolor=COR_FUNDO, font=dict(color=COR_TEXTO),
    margin=dict(l=10, r=10, t=10, b=80),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    separators=",.",
    transition_duration=0  
)
fig.update_traces(orientation="v", selector=dict(type="bar"))

fig.update_xaxes(
    type="category",
    categoryorder="array",
    categoryarray=list(agg["Data_Label"]),
    tickangle=55, gridcolor=COR_GRADE, zeroline=False
)
fig.update_yaxes(
    title_text="Valor a Vencer (R$)", secondary_y=False,
    gridcolor=COR_GRADE, zeroline=False, tickformat=",.0f", tickprefix="R$ "
)
fig.update_yaxes(
    title_text="Quantidade de Clientes com Vencimento", secondary_y=True,
    gridcolor=COR_GRADE, zeroline=False, dtick=1, tick0=0, rangemode="tozero"
)

st.subheader("Valor Total a Vencer (R$) e Quantidade de Clientes por Data")
st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})

# seletor de data 
if not agg.empty:
    default_idx = 0
    sel_label = st.selectbox(
        "Selecione a data de análise",
        options=agg["Data_Label"].tolist(),
        index=default_idx
    )
else:
    sel_label = None

# kpis
k1, k2 = st.columns(2)
if sel_label in label2date:
    linha = agg.loc[agg["Data_Label"].eq(sel_label)].iloc[0]
    k1_box = kpi_box("Quantidade de Clientes",
                     f"{int(linha['Clientes_do_Dia'])}")
    k2_box = kpi_box("Valor Total a Vencer (R$)",
                     brl(linha["Valor_do_Dia"]))
else:
    k1_box = kpi_box("Quantidade de Clientes", "0")
    k2_box = kpi_box("Valor Total a Vencer (R$)", brl(0))

# tabela de clientes
st.subheader("Clientes com COEs vencendo")
if sel_label in label2date:
    sel_date = label2date[sel_label]
    df_dia = df.loc[df["Data_Vencimento"].eq(sel_date)].copy()
    
    grp_cols = ["Nome_Cliente"]
    if "Numero_Conta" in df_dia.columns:
        det = (
            df_dia.groupby(grp_cols, as_index=False)
                  .agg(
                      Contas_Distintas=("Numero_Conta", "nunique"),
                      Qtde_COEs=("Valor_Atual", "size"),
                      Valor_Aplicado=("Valor_Aplicado", "sum"),
                      Valor_Atual=("Valor_Atual", "sum"),
                      Tenor_Medio_dias=("Tenor_dias", "mean"),
                  )
                  .sort_values("Valor_Atual", ascending=False)
        )
    else:
        det = (
            df_dia.groupby(grp_cols, as_index=False)
                  .agg(
                      Qtde_COEs=("Valor_Atual", "size"),
                      Valor_Aplicado=("Valor_Aplicado", "sum"),
                      Valor_Atual=("Valor_Atual", "sum"),
                      Tenor_Medio_dias=("Tenor_dias", "mean"),
                  )
                  .sort_values("Valor_Atual", ascending=False)
        )

    det["Valor_Aplicado"] = det["Valor_Aplicado"].map(brl)
    det["Valor_Atual"]    = det["Valor_Atual"].map(brl)
    det["Tenor_Medio_dias"] = det["Tenor_Medio_dias"].round(1)

    det = det.rename(columns={
        "Nome_Cliente": "Nome do Cliente",
        "Qtde_COEs": "Qtde de COEs no Dia",
        "Valor_Aplicado": "Valor Aplicado (R$)",
        "Valor_Atual": "Valor Atual (R$)",
    })

    st.dataframe(det, width="stretch", hide_index=True)
else:
    st.info("Selecione uma data acima para ver os clientes do dia.")

# ml da base
st.markdown("---")
st.header("Anomalias e Segmentação na base com Machine Learning")

# explicacao 
with st.expander("O que significam as métricas usadas?"):
    st.markdown(
        """
- **Total_Valor_Atual**: soma do valor atual de coes por cliente na base inteira  
- **Qtde_COEs**: quantidade de linhas de coes do cliente na base  
- **Contas_Distintas**: numero de contas diferentes associadas ao cliente  
- **Ticket_Medio**: valor medio por coe do cliente  
- **Tenor_Medio**: prazo medio em dias entre aplicacao e vencimento  
        """
    )

if "Nome_Cliente" in df.columns and not df.empty:
    feats = (
        df.groupby("Nome_Cliente", as_index=False)
          .agg(
              Total_Valor_Atual=("Valor_Atual", "sum"),
              Qtde_COEs=("Valor_Atual", "size"),
              Contas_Distintas=("Numero_Conta", "nunique") if "Numero_Conta" in df.columns else ("ID_Cliente", "nunique"),
              Ticket_Medio=("Valor_Atual", "mean"),
              Tenor_Medio=("Tenor_dias", "mean"),
          )
    )
    # trata valores inf e nan
    feats = feats.replace([np.inf, -np.inf], np.nan).fillna(0)
    feats = feats.sort_values("Total_Valor_Atual", ascending=False).reset_index(drop=True)

    with st.expander("Ver base de features por cliente", expanded=False):
        tmp = feats.copy()
        tmp["Total_Valor_Atual"] = tmp["Total_Valor_Atual"].map(brl)
        tmp["Ticket_Medio"] = tmp["Ticket_Medio"].map(brl)
        tmp["Tenor_Medio"] = tmp["Tenor_Medio"].round(1)
        st.dataframe(tmp, width="stretch", hide_index=True)

    #  modelos
    try:
        from sklearn.preprocessing import StandardScaler
        from sklearn.ensemble import IsolationForest
        from sklearn.cluster import KMeans

        # anomalias com isolation forest 
        # usa as colunas numericas abaixo para aprender o padrao do grupo
        num_cols = ["Total_Valor_Atual", "Qtde_COEs", "Ticket_Medio", "Tenor_Medio", "Contas_Distintas"]
        X = feats[num_cols].astype(float).values

        if len(feats) >= 8 and np.isfinite(X).all():
            # padroniza as features para media 0 e desvio 1
            scaler = StandardScaler()
            Xs = scaler.fit_transform(X)

            # treina o modelo de anomalias
            iso = IsolationForest(
                n_estimators=300,
                contamination="auto",
                random_state=42
            )
            iso.fit(Xs)

            # predicoes -1 anomalia, 1 normal
            pred = iso.predict(Xs)

            # score_samples negativo quanto menor, mais normal
            score = -iso.score_samples(Xs)

            feats["Anomalia"] = np.where(pred == -1, "Sim", "Não")
            feats["Score_Anomalia"] = score

            # mostra apenas anomalias ordenadas pela gravidade
            anom = feats.loc[feats["Anomalia"] == "Sim"].copy()
            anom = anom.sort_values("Score_Anomalia", ascending=False)

            st.subheader("Anomalias na base")
            if not anom.empty:
                view = anom[["Nome_Cliente","Anomalia","Score_Anomalia","Total_Valor_Atual","Qtde_COEs","Ticket_Medio","Tenor_Medio","Contas_Distintas"]].copy()
                view["Total_Valor_Atual"] = view["Total_Valor_Atual"].map(brl)
                view["Ticket_Medio"] = view["Ticket_Medio"].map(brl)
                view["Tenor_Medio"] = view["Tenor_Medio"].round(1)
                view["Score_Anomalia"] = view["Score_Anomalia"].round(3)
                st.dataframe(view, width="stretch", hide_index=True)
                st.caption("Clientes marcados como anomalia tendem a ter tickets, quantidades ou prazos fora do padrão do grupo.")
            else:
                st.info("Nenhuma anomalia relevante detectada na base inteira.")
        else:
            st.info("Poucos clientes para detecção de anomalias na base inteira (mínimo sugerido: 8).")

        # segmentacao com k-means 
        # cria grupos de clientes com padroes semelhantes para facilitar a estrategia comercial
        if len(feats) >= 4 and np.isfinite(X).all():
            scaler2 = StandardScaler()
            Xs2 = scaler2.fit_transform(X)

            # define numero de clusters via heuristica simples
            k = min(4, max(2, len(feats)//5))
            km = KMeans(n_clusters=k, n_init=10, random_state=42)
            feats["Cluster"] = km.fit_predict(Xs2)

            # resumo por cluster
            resumo = (
                feats.groupby("Cluster", as_index=False)
                     .agg(
                         Clientes=("Nome_Cliente","count"),
                         Valor_Total=("Total_Valor_Atual","sum"),
                         Ticket_Medio=("Ticket_Medio","mean"),
                         Tenor_Medio=("Tenor_Medio","mean"),
                     )
                     .sort_values("Valor_Total", ascending=False)
            )
            viewr = resumo.copy()
            viewr["Valor_Total"] = viewr["Valor_Total"].map(brl)
            viewr["Ticket_Medio"] = viewr["Ticket_Medio"].map(brl)
            viewr["Tenor_Medio"] = viewr["Tenor_Medio"].round(1)

            st.subheader("Segmentos de clientes na base inteira")
            st.dataframe(viewr, width="stretch", hide_index=True)

            # detalhamento por cliente com cluster
            with st.expander("Ver clientes e seus clusters", expanded=False):
                v = feats[["Nome_Cliente","Cluster","Total_Valor_Atual","Qtde_COEs","Ticket_Medio","Tenor_Medio","Contas_Distintas"]].copy()
                v["Total_Valor_Atual"] = v["Total_Valor_Atual"].map(brl)
                v["Ticket_Medio"] = v["Ticket_Medio"].map(brl)
                v["Tenor_Medio"] = v["Tenor_Medio"].round(1)
                st.dataframe(v.sort_values(["Cluster","Total_Valor_Atual"], ascending=[True, False]),
                             width="stretch", hide_index=True)
        else:
            st.info("Poucos clientes para segmentação na base inteira (mínimo sugerido: 4).")

    except Exception as e:
        st.error(
            "A seção de ML requer scikit-learn. "
            "Instale com: `pip install scikit-learn`.\n\n"
            f"Detalhes: {e}"
        )
else:
    st.info("Base vazia ou sem coluna 'Nome_Cliente' para análise de ML.")

