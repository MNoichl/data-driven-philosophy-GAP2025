

#
![](static_images/max_noichl_eu_talk.png){width=40% fig-align="center"}

:::{.r-stack}
To follow along, visit [www.maxnoichl.eu/talk](www.maxnoichl.eu/talk)
:::


# About Me

- Promovendus at **Utrecht University**
- Interested in computational methods for philosophy.
- Background in HPS, education in comp. methods via GESIS, CompSciHub Vienna, SFI.
- Working on cognitive dynamics of philosophy, scientific model transfer, scientific information flow...


# Goals of this talk 

- Introduce key concepts of network science
- Show some examples from philosophy
- Share resources

# What are networks?

1. Networks are graphs: [Network Vocabulary](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/intro_graphics_01.png"}
2. Networks are a data structure: [Networks as data](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/intro_graphics_02.png"}
    - Many things can be understood as networks that we don't immediately think of as such. E. g.  bimodal word – document networks, similarity based networks based on embedding models, networks of collocations, etc. 
3. Networks are Matrices: [Adjacency Matrix](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/intro_graphics_03.png"}
<!-- affinity matrix
Interdependence -->


# {background-iframe="background_network.html" background-size="120%" background-position="left bottom" background-opacity="1."}
::: {.absolute top=-50 left=700 background="#000" .title-block}
<h1 class="title">Some exciting results!</h1>
:::


# Some exciting results!

* Scale free networks.
* Strength of weak ties. 

# Degree distributions

- Let's generate simple random graph: [Erdős–Rényi model](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/er_construct.gif"}
- Let's look at the degree distribution: [ER-degrees](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/er_hist.png"}
- But it turns out, many real world networks don't look like this! [Example: Citations in philosophy](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url='images/citations_distribution.png'}


# Scale free networks

- Let's try *Preferential attachement*:  [Barabási-Albert Model](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/ba_construct.gif"}
- The distribution now follows a power-law $m * k^{- \alpha}$:  [Degree distribution BA-model](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/ba_hist.png"}
- It's common to look at these distributions on log-log plot:  [Let's compare!](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/ba_er_ccdf.png"}
- The powerlaw exhibits *scale-freeness*: However you zoom around the graph, it looks the same.
- Also, power-law distributions are *heavy-tailed*: [Tail-comparison](#){.fragment .opens-modal
  data-modal-type="iframe"
  data-modal-url="https://www.desmos.com/calculator/v9lrucx3ea"}
- They are to robust to random, but fragile to targeted attacks: The average node doesn't predict behaviour.
- These distributions are very common, and suggest rich-get-richer processes: [The Matthew effect](#){.fragment .opens-modal
  data-modal-type="html"
  data-modal-url="#somehiddendiv"}

::: {#somehiddendiv .hidden}
For to everyone who has, more will be given, and he will have abundance; but from him who does not have, even what he has will be taken away.

— Matthew 25:29
:::


<br>

::: aside
@barabasiEmergenceScalingRandom1999; 
:::

# Scale free networks

- These distributions are also common in social networks in philosophy: [Petrovich, 2021](#){.fragment .opens-modal
  data-modal-type="iframe"
  data-modal-url="images/petrovich_2021.pdf"}
- And historically: [Collins, 1998 (p. 56/80)](#){.fragment .opens-modal
  data-modal-type="iframe"
  data-modal-url="images/Collins_1998.pdf"}
- Also useful to philosophy of science: [Rubin & Schneider, 2021](#){.fragment .opens-modal
  data-modal-type="iframe"
  data-modal-url="images/Rubin_Schneider_2021_Priority and privilege in scientific discovery.pdf"}

<br>

::: aside
@collinsSociologyPhilosophiesGlobal1998; @petrovichAcknowledgmentsbasedNetworksMapping2022; @rubinPriorityPrivilegeScientific2021
:::

# Small worlds & weak ties

- We start with a simple [random geometric graph.](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/img1_geometric_graph.png"}
- A signal takes quite a while to propagate: [Signal propagation.](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/anim1_signal.gif"}
- But if we add just very few long connections: ['weak' ties: ](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/img2_with_long_ties.png"}
-  The signal now propagates much faster: [Strength of weak ties.](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/anim2_signal_with_long_ties.gif"}
- Weak, very distant ties are crucial for information spread in real communities. Many real world networks exhibit "small world effects". (Formally: High clustering and small shortest path length.)
- These properties are highly relevant to network epistemology: [Zollman, 2009](#){.fragment .opens-modal
  data-modal-type="iframe"
  data-modal-url="images/Zollman - 2010.pdf"}

<br>

::: aside
@zollmanEpistemicBenefitTransient2009, @granovetterStrengthWeakTies1983; @wattsCollectiveDynamicsSmallworld1998
:::


# {background-iframe="background_network.html" background-size="120%" background-position="left bottom" background-opacity="1."}
::: {.absolute top=-50 left=700 background="#000" .title-block}
<h1 class="title">Exploring networks!</h1>
:::


# Visualization
- Force directed networks: [Explore in gephi-lite](#){.fragment .opens-modal
  data-modal-type="iframe"
  data-modal-url="https://gephi.org/gephi-lite/"}
- Some guidelines, [by Mathieu Jacomy.](#){.fragment .opens-modal
  data-modal-type="iframe"
  data-modal-url="https://reticular.hypotheses.org/"}
- UMAP/t-SNE:  [Flattening mammoths](#){.fragment .opens-modal
  data-modal-type="iframe"
  data-modal-url="https://pair-code.github.io/understanding-umap/"}
- Edge bundling:  [Helpful, but handle with caution.](#){.fragment .opens-modal
  data-modal-type="iframe"
  data-modal-url="https://schochastics.github.io/edgebundle/"}



# Clustering, Community-detection

- What's a good clustering? 
- One idea: Maximize Modularity (more edges within cluster than expected by a null-model)
- Louvain-algorithm: [First Iteration.](#){.fragment .opens-modal
  data-modal-type="video"
  data-modal-url="images/louvain_stage_0.mp4"}
- We then merge clusters & repeat: [Second iteration.](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/louvain_transition_0_to_1.png"}
- Leiden-algorithm (slightly improved version of Louvain)
- Stochastic Blockmodel:  [Explicit statistical model of communities.](#){.fragment .opens-modal
  data-modal-type="iframe"
  data-modal-url="https://graph-tool.skewed.de/static/docs/stable/demos/inference/inference.html"}
  
::: {.notes}
- Null = configuration model, preserves degree
- Each node in its own cluster.
- Greedy algo: Does modularity improve? keep, else reject.
:::


# Some general thoughts

- Hairballs are the start of an analysis, not the end.
- Are you interested in network effects? Or do you want to control for them?
- *Don't accidentally rediscover random graph theory!*
- (Null models are important!)

# ERGMS

- How to incorporate null-models into graph-analysis? – There are so many possible graphs!
- One option: *Exponential Random Graph Models* fitted using MCMC to the adequate graph-category.
- Are features like density, star-patterns, homophilic connections, etc. more common than expected under chance?
- Used to investigate Philosophy of Science itself: [McLevey et al., 2018](#){.fragment .opens-modal
  data-modal-type="iframe"
  data-modal-url="images/mclevey2018.pdf"}
- Hard to make work in practice, identifiability problems.



# {background-iframe="background_network.html" background-size="120%" background-position="left bottom" background-opacity="1."}
::: {.absolute top=-50 left=700 background="#000" .title-block}
<h1 class="title">Tools</h1>
:::


# Software

* Network analysis in Python: [networkx](#){.opens-modal .fragment 
  data-modal-type="iframe"
  data-modal-url="https://networkx.org/documentation/stable/"}
* Advanced network modelling in python: [graph-tool](#){.opens-modal .fragment 
  data-modal-type="iframe"
  data-modal-url="https://graph-tool.skewed.de/"}
* Network-analysis in  R (also includes ERGMs): [statnet](#){.opens-modal .fragment 
  data-modal-type="iframe"
  data-modal-url="https://statnet.org/"}
* Interactive network-visualisation: [Gephi](#){.opens-modal .fragment 
  data-modal-type="iframe"
  data-modal-url="https://gephi.org/"}
* Easy exploration of scientometrics: [Vos-Viewer](#){.opens-modal .fragment 
  data-modal-type="iframe"
  data-modal-url="https://app.vosviewer.com/docs/"}
* Prettier network-plots in python: [Pylabeladjust](#){.opens-modal .fragment 
  data-modal-type="iframe"
  data-modal-url="https://pypi.org/project/pylabeladjust/"}

# {background-iframe="background_network.html" background-size="120%" background-position="left bottom" background-opacity="1."}
::: {.absolute top=-50 left=700 background="#000" .title-block}
<h1 class="title">Data sources</h1>
:::


# Citation data
- Web-of-Science
- CrossRef (Great to fix bad citations.)
- Open-Alex (also great for abstracts, OA-literature)

# Text-parsing

- Jstor 
- Preprint servers (See ArxivHTML)
- Texts can be network-data! [Malaterre & Lareau, 2022 ](#){.fragment .opens-modal
  data-modal-type="iframe"
  data-modal-url="images/Malaterre und Lareau - 2022 - The early days of contemporary philosophy of scien.pdf"}

<br>

::: aside
@malaterreEarlyDaysContemporary2022
:::



# {background-iframe="background_network.html" background-size="120%" background-position="left bottom" background-opacity="1."}
::: {.absolute top=-50 left=700 background="#000" .title-block}
<h1 class="title">Some of my own work</h1>
:::



# Modeling Model-transfer

- OpenAlex Mapper: [A huggingface space.](#){.fragment .opens-modal
  data-modal-type="iframe"
  data-modal-url="https://maxnoichl-openalex-mapper.hf.space"}
- Useful to investigate model-transfer, concept-spread,   [scientific cultures.](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/random_forest_logistic_regression_distribution_base_plot.png"}

# Cognitive Dynamics of philosophy

- Philosophical literature, uniform sample, 1950-2015
- We parse positions & examples: [Some initial visualizations.](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/positions_examples.png"}
- Some first results (very wip  ): [Predicting Citation counts](#){.fragment .opens-modal
  data-modal-type="image"
  data-modal-url="images/citiaon_model_log_citations_ols_features_coefficients.png"}

# {background-iframe="background_network.html" background-size="120%" background-position="left bottom" background-opacity="1."}
::: {.absolute top=-50 left=700 background="#000" .title-block}
<h1 class="title">Thank you!</h1>
:::


# Literature




