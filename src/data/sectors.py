"""
Static sector mapping for universe stocks.

Simplified GICS-like sectors. Used for:
  - Candidate diversity analysis
  - Sector heat calculation in market summary

CONFIG: Edit the dicts below to update sector assignments.
"""

# US (NASDAQ-100 subset)
US_SECTORS = {
    "AAPL.US": "Technology", "MSFT.US": "Technology", "NVDA.US": "Technology",
    "GOOGL.US": "Technology", "AMZN.US": "Consumer", "META.US": "Technology",
    "TSLA.US": "Consumer", "AVGO.US": "Technology", "AMD.US": "Technology",
    "ADBE.US": "Technology", "CRM.US": "Technology", "INTC.US": "Technology",
    "QCOM.US": "Technology", "TXN.US": "Technology", "MU.US": "Technology",
    "AMAT.US": "Technology", "LRCX.US": "Technology", "KLAC.US": "Technology",
    "MRVL.US": "Technology", "NXPI.US": "Technology", "MCHP.US": "Technology",
    "ON.US": "Technology", "ADI.US": "Technology", "SNPS.US": "Technology",
    "CDNS.US": "Technology", "NXPI.US": "Technology", "ARM.US": "Technology",
    "ASML.US": "Technology", "APP.US": "Technology", "SHOP.US": "Technology",
    "MSTR.US": "Technology", "COIN.US": "Technology",
    "NFLX.US": "Communication", "DIS.US": "Communication", "TMUS.US": "Communication",
    "CMCSA.US": "Communication", "CHTR.US": "Communication",
    "JPM.US": "Financial", "V.US": "Financial", "MA.US": "Financial",
    "BAC.US": "Financial", "GS.US": "Financial", "MS.US": "Financial",
    "BLK.US": "Financial", "SCHW.US": "Financial", "AXP.US": "Financial",
    "PGR.US": "Financial", "MMC.US": "Financial",
    "UNH.US": "Healthcare", "JNJ.US": "Healthcare", "LLY.US": "Healthcare",
    "PFE.US": "Healthcare", "ABBV.US": "Healthcare", "MRK.US": "Healthcare",
    "TMO.US": "Healthcare", "ABT.US": "Healthcare", "DHR.US": "Healthcare",
    "BMY.US": "Healthcare", "AMGN.US": "Healthcare", "GILD.US": "Healthcare",
    "ISRG.US": "Healthcare", "VRTX.US": "Healthcare", "REGN.US": "Healthcare",
    "ALNY.US": "Healthcare", "INSM.US": "Healthcare", "BIIB.US": "Healthcare",
    "IDXX.US": "Healthcare",
    "PG.US": "Consumer", "KO.US": "Consumer", "PEP.US": "Consumer",
    "COST.US": "Consumer", "WMT.US": "Consumer", "MCD.US": "Consumer",
    "NKE.US": "Consumer", "SBUX.US": "Consumer", "TJX.US": "Consumer",
    "BKNG.US": "Consumer", "ABNB.US": "Consumer", "ORLY.US": "Consumer",
    "CMG.US": "Consumer",
    "XOM.US": "Energy", "CVX.US": "Energy", "COP.US": "Energy",
    "SLB.US": "Energy", "EOG.US": "Energy",
    "NEE.US": "Utilities", "DUK.US": "Utilities", "SO.US": "Utilities",
    "D.US": "Utilities", "AEP.US": "Utilities", "SRE.US": "Utilities",
    "CEG.US": "Utilities", "PCG.US": "Utilities",
    "CAT.US": "Industrial", "DE.US": "Industrial", "HON.US": "Industrial",
    "UNP.US": "Industrial", "UPS.US": "Industrial", "RTX.US": "Industrial",
    "BA.US": "Industrial", "GE.US": "Industrial", "LMT.US": "Industrial",
    "PCAR.US": "Industrial", "WM.US": "Industrial",
    "PLTR.US": "Technology", "PANW.US": "Technology", "CRWD.US": "Technology",
    "ZS.US": "Technology", "DDOG.US": "Technology", "NET.US": "Technology",
    "MDB.US": "Technology", "SNOW.US": "Technology",
    "PYPL.US": "Financial", "SQ.US": "Financial",
    "MRNA.US": "Healthcare", "DXCM.US": "Healthcare",
    "CSX.US": "Industrial", "FAST.US": "Industrial",
    "GFS.US": "Technology", "WDC.US": "Technology", "STX.US": "Technology",
    "SNDK.US": "Technology",
    "VRSK.US": "Industrial", "CTSH.US": "Technology",
    "FANG.US": "Energy", "HAL.US": "Energy", "OXY.US": "Energy",
    "MPC.US": "Energy", "PSX.US": "Energy",
    "CPRT.US": "Industrial", "ODFL.US": "Industrial",
    "PAYX.US": "Industrial", "ANSS.US": "Technology",
    "EXC.US": "Utilities", "XEL.US": "Utilities",
    "WBD.US": "Communication", "EA.US": "Communication",
    "TTWO.US": "Communication",
    "KHC.US": "Consumer", "MNST.US": "Consumer",
    "VRSN.US": "Technology",
    "BKR.US": "Energy",
    "A.US": "Healthcare",
    "GEHC.US": "Healthcare",
    "CARR.US": "Industrial", "OTIS.US": "Industrial",
    "CTAS.US": "Industrial", "ROST.US": "Consumer",
    "DLTR.US": "Consumer",
    "AZN.US": "Healthcare",
    "SIRI.US": "Communication",
    "TEAM.US": "Technology",
    "LCID.US": "Consumer", "RIVN.US": "Consumer",
    "SMCI.US": "Technology",
    "AXON.US": "Industrial",
    "TTD.US": "Technology",
    "HOOD.US": "Financial",
    "DASH.US": "Consumer",
    "AB.US": "Financial",
    "CEG.US": "Utilities",
    "DECK.US": "Consumer",
    "VST.US": "Utilities",
    "TSCO.US": "Consumer",
    "KDP.US": "Consumer",
    "ONC.US": "Healthcare",
    "ARGX.US": "Healthcare",
}

# HK (HSI subset)
HK_SECTORS = {
    "0700.HK": "Technology", "9988.HK": "Technology", "9999.HK": "Technology",
    "3690.HK": "Technology", "1810.HK": "Technology", "0268.HK": "Technology",
    "0241.HK": "Technology", "0992.HK": "Technology", "2018.HK": "Technology",
    "0285.HK": "Technology",
    "0005.HK": "Financial", "0011.HK": "Financial", "0388.HK": "Financial",
    "1299.HK": "Financial", "2318.HK": "Financial", "2628.HK": "Financial",
    "1398.HK": "Financial", "0939.HK": "Financial", "3988.HK": "Financial",
    "0001.HK": "Industrial", "0002.HK": "Industrial", "0003.HK": "Industrial",
    "0006.HK": "Industrial", "0012.HK": "Industrial", "0016.HK": "Industrial",
    "0017.HK": "Industrial", "0027.HK": "Industrial", "0066.HK": "Industrial",
    "0101.HK": "Industrial", "0175.HK": "Industrial", "0267.HK": "Industrial",
    "0288.HK": "Industrial", "0291.HK": "Industrial", "0300.HK": "Industrial",
    "0316.HK": "Industrial", "0322.HK": "Industrial", "0386.HK": "Energy",
    "0669.HK": "Industrial", "0688.HK": "Technology", "0762.HK": "Communication",
    "0823.HK": "Industrial", "0857.HK": "Energy", "0868.HK": "Industrial",
    "0881.HK": "Industrial", "0883.HK": "Energy", "0916.HK": "Industrial",
    "0941.HK": "Communication", "0960.HK": "Industrial", "0968.HK": "Industrial",
    "0981.HK": "Technology", "1024.HK": "Technology", "1038.HK": "Industrial",
    "1044.HK": "Industrial", "1093.HK": "Healthcare", "1109.HK": "Industrial",
    "1113.HK": "Industrial", "1177.HK": "Healthcare", "1211.HK": "Consumer",
    "1347.HK": "Technology", "1378.HK": "Consumer", "1398.HK": "Financial",
    "1876.HK": "Consumer", "1928.HK": "Consumer", "1929.HK": "Consumer",
    "1997.HK": "Industrial", "2007.HK": "Industrial", "2013.HK": "Industrial",
    "2015.HK": "Consumer", "2020.HK": "Consumer", "2269.HK": "Healthcare",
    "2313.HK": "Consumer", "2319.HK": "Consumer", "2331.HK": "Consumer",
    "2382.HK": "Technology", "2388.HK": "Financial", "2518.HK": "Technology",
    "3323.HK": "Industrial", "3692.HK": "Healthcare", "3968.HK": "Financial",
    "6098.HK": "Industrial", "6618.HK": "Technology", "6690.HK": "Technology",
    "6862.HK": "Consumer", "9618.HK": "Technology", "9626.HK": "Technology",
    "9633.HK": "Consumer", "9698.HK": "Technology", "9868.HK": "Consumer",
    "9888.HK": "Technology", "9901.HK": "Technology", "9961.HK": "Consumer",
    "9988.HK": "Technology", "9992.HK": "Technology",
    "9999.HK": "Technology",
}

# CN (SSE 50)
CN_SECTORS = {
    "sh.600519": "Consumer", "sh.601318": "Financial", "sh.600036": "Financial",
    "sh.600900": "Utilities", "sh.601398": "Financial", "sh.600276": "Healthcare",
    "sh.600031": "Industrial", "sh.601166": "Financial", "sh.600030": "Financial",
    "sh.601668": "Industrial", "sh.600887": "Consumer", "sh.601888": "Consumer",
    "sh.600809": "Consumer", "sh.600690": "Consumer", "sh.601012": "Healthcare",
    "sh.600585": "Industrial", "sh.601601": "Financial", "sh.600048": "Industrial",
    "sh.601211": "Consumer", "sh.600028": "Energy", "sh.601857": "Energy",
    "sh.600016": "Financial", "sh.601328": "Financial", "sh.600019": "Materials",
    "sh.600050": "Communication", "sh.601688": "Financial", "sh.600089": "Industrial",
    "sh.600104": "Consumer", "sh.600111": "Materials", "sh.600150": "Industrial",
    "sh.600183": "Technology", "sh.600309": "Materials", "sh.600406": "Industrial",
    "sh.600760": "Technology", "sh.603019": "Technology", "sh.603993": "Materials",
    "sh.688008": "Technology", "sh.688012": "Technology", "sh.688111": "Healthcare",
    "sh.688256": "Technology", "sh.688981": "Healthcare",
    "sh.601600": "Industrial", "sh.601899": "Materials",
}

# Crypto
CRYPTO_SECTORS = {
    "BTC-USD.CC": "Crypto", "ETH-USD.CC": "Crypto", "BNB-USD.CC": "Crypto",
    "SOL-USD.CC": "Crypto", "XRP-USD.CC": "Crypto", "ADA-USD.CC": "Crypto",
    "DOGE-USD.CC": "Crypto", "DOT-USD.CC": "Crypto", "AVAX-USD.CC": "Crypto",
    "LINK-USD.CC": "Crypto", "MATIC-USD.CC": "Crypto", "UNI-USD.CC": "Crypto",
    "SHIB-USD.CC": "Crypto", "LTC-USD.CC": "Crypto", "ATOM-USD.CC": "Crypto",
    "XLM-USD.CC": "Crypto", "TRX-USD.CC": "Crypto", "FIL-USD.CC": "Crypto",
    "ETC-USD.CC": "Crypto", "ARB-USD.CC": "Crypto",
}


def get_sector(ticker: str) -> str:
    """Get sector for a ticker. Returns empty string if unknown."""
    return (
        US_SECTORS.get(ticker)
        or HK_SECTORS.get(ticker)
        or CN_SECTORS.get(ticker)
        or CRYPTO_SECTORS.get(ticker)
        or ""
    )
