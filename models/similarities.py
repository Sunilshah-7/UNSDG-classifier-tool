from sentence_transformers import SentenceTransformer

_embedder = None
def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return _embedder


SDG_DESCS = ["""SDG 1 : extreme poverty income below dollar threshold cash transfers 
    social protection floors universal basic coverage poorest vulnerable 
    populations economic resources land property inheritance microfinance 
    disaster risk resilience eradicate destitution livelihood safety nets
    proportion population poverty line consumption expenditure""",

    """SDG2 : hunger food insecurity malnutrition stunting wasting underweight 
    smallholder farmers agricultural productivity crop yields soil fertility 
    seed banks genetic diversity food systems rural markets price volatility 
    famine emergency food aid nutrition programmes dietary diversity 
    sustainable agriculture irrigation agroecology""",

    """SDG 3 : health maternal mortality neonatal child under-five deaths communicable disease 
    HIV AIDS tuberculosis malaria hepatitis vaccines immunization epidemic 
    non-communicable cardiovascular cancer diabetes mental health substance abuse 
    universal health coverage medicines essential drugs reproductive health 
    family planning skilled birth attendance""",

    """SDG 4 : quality education primary secondary school completion literacy numeracy dropout rates 
    early childhood education vocational training technical skills TVET 
    higher education scholarships qualified teachers learning outcomes 
    disability inclusive education gender parity enrolment attendance 
    digital literacy foundational skills""",

    """SDG 5 : gender equality women empowerment discrimination violence against women 
    female genital mutilation child marriage unpaid domestic care work 
    equal pay wage gap sexual reproductive rights contraception 
    women leadership political participation land ownership property rights 
    girls education menstrual hygiene""",

    """SDG 6 : drinking water safe sanitation open defecation WASH hygiene handwashing 
    water quality treatment wastewater recycling water use efficiency 
    water scarcity transboundary river basin groundwater aquifer 
    water infrastructure rural piped supply irrigation systems 
    water borne disease cholera dysentery""",

    """SDG 7 : electricity access clean cooking fuels renewable energy solar wind 
    hydropower geothermal biomass energy efficiency appliances buildings 
    fossil fuel subsidy reform grid infrastructure off grid mini grid 
    energy poverty kerosene candles kilowatt megawatt gigawatt 
    carbon emission energy transition""",

    """SDG 8 : decent work employment job creation unemployment youth labour 
    forced labour modern slavery human trafficking child labour 
    informal economy living wage productivity GDP growth 
    small medium enterprises entrepreneurship financial inclusion 
    banking credit microfinance tourism sustainable business""",

    """SDG 9 : infrastructure roads bridges ports railways broadband internet 
    connectivity ICT access rural urban industrialization manufacturing 
    value added industry research development innovation patents 
    technology transfer mobile network 4G 5G fintech 
    resilient infrastructure disaster proof construction""",

    """SDG 10 :income inequality gini coefficient wealth distribution top bottom 
    decile palma ratio social mobility discrimination race ethnicity 
    disability migrants remittances immigration policy 
    developing country representation voting rights 
    progressive taxation redistribution fiscal policy""",

    """SDG 11 : urban cities slums informal settlements affordable housing 
    public transport sustainable mobility pedestrian cycling 
    air pollution particulate matter urban planning zoning 
    cultural heritage disaster risk reduction flood resilience 
    green spaces parks waste collection municipal solid waste""",

    """SDG 12 : sustainable consumption production lifecycle footprint 
    toxic chemicals hazardous waste electronic waste e-waste 
    circular economy recycling repair reuse reduce 
    food waste loss supply chain corporate sustainability reporting 
    consumer awareness green procurement public spending""",

    """SDG 13 : climate change greenhouse gas emissions carbon dioxide methane 
    global warming temperature rise adaptation mitigation 
    climate resilience extreme weather drought flood cyclone 
    Paris agreement nationally determined contributions NDC 
    climate finance loss damage early warning systems""",

    """SDG 14 : ocean marine coastal fisheries overfishing illegal unreported 
    coral reef seagrass mangrove wetland biodiversity 
    plastic pollution marine debris ocean acidification 
    deep sea mining small island developing states 
    exclusive economic zone aquaculture blue economy""",

    """SDG 15 : terrestrial forest deforestation land degradation desertification 
    biodiversity species extinction poaching trafficking wildlife 
    protected areas national park conservation habitat restoration 
    invasive species mountain ecosystem dryland 
    land tenure soil carbon sequestration reforestation""",

    """SDG 16 : peace conflict violence armed groups rule of law justice 
    access to legal aid courts accountable institutions 
    corruption bribery transparency freedom of information 
    press freedom civil society participation human rights 
    birth registration identity documents stateless persons 
    illicit financial flows money laundering tax evasion""",

    """SDG 17 : global partnership development finance ODA official aid 
    debt relief developing countries technology transfer capacity building 
    trade WTO multilateral system policy coherence 
    data statistics monitoring reporting voluntary national review 
    south south cooperation multi stakeholder blended finance"""]