# Expérience 2 — traçabilité historique des formules et du protocole

> **Archive historique, non statut actuel.** Les sections datées du 23 août
> ci-dessous décrivent le développement avant exécution finale. Les runs finaux
> ont ensuite été exécutés. Le contrôle courant v40 est `MATH_V40.md`, les
> résultats d'exécution sont dans `VALIDATION_REPORT.md`. Les ordres affichés
> sont désormais 1.1/1.4 (post-traitement), non 1/1.25. Les anciens chemins,
> hashs et numéros de ligne ne sont pas ceux de la distribution actuelle.

> **Statut révisé : 23 août 2026.** Les formules ont été revalidées contre le
> manuscrit autoritatif `mainaistats_no_A2 (2).tex`. Le pilote courant a été
> exécuté et sa porte scientifique a passé ; le niveau final reste prospectif et
> n'a pas été exécuté. Les anciens numéros de lignes v22 conservés plus bas sont
> historiques ; les équations physiques correspondantes sont inchangées dans
> l'autorité courante.

## Sources autoritatives

- Execution-time manuscript snapshot: **mainaistats_no_A2 (2).tex**, SHA-256
  **A2CEB81155566B43030677C90F5A62437146449118DACC17F0EC2B963DCE56BF**.
- Spécification indépendante courante : **reports/phase0_independent_spec_latest.md**,
  SHA-256 **5EC54D27E8165CD164173619CA52EE2C2D04B193536155CDFA67FFF8BDBED838**.
- Smoke : **configs/smoke/exp2.toml**, SHA-256
  **C2ACDD23F29FB4BF61E9C825CF1BF26EF44C5168CF909C48D89ADC7B952066A5**.
- Pilot : **configs/pilot/exp2.toml**, SHA-256
  **044D11CC6DC4A907A58329FCD6C65D264A63C7F5D60FA9D9C247ABC2C2EEEF79**.
- Final : **configs/final/exp2.toml**, SHA-256
  **15D41F9A7AF8E38039E516AD93BADC3486C2604F7B8C3A4E6079C984C5993752**.

L’autorité est hiérarchisée ainsi : équations du manuscrit courant, protocole Exp2 figé dans
les TOML, puis implémentation. Les choix propres à Exp2 — Student standard,
grilles, tailles d’échantillon et métriques — ne sont pas présentés comme des
énoncés du manuscrit.

## Verdict de raccord avec l'autorité courante

Exp2 est une nouvelle expérience d’erreur d’initialisation, plus directe que le
couplage synchrone actuellement décrit pour la Figure dynamique (c). Dans v22,
ce panneau mesure une moyenne de distances trajectorielles couplées, qui est une
borne supérieure de \(W_1\) (lignes 2515–2534). Exp2 mesure au contraire des
distances de Wasserstein empiriques unidimensionnelles entre des lois simulées
indépendamment et des marginales forward exactes.

Cette distinction doit être conservée si Exp2 est intégrée au manuscrit : elle
ne constitue pas silencieusement une « régénération » du protocole de la Figure
dynamique (c). Elle peut servir de validation complémentaire, ou remplacer ce
panneau après décision éditoriale. La remarque de v22 signalant l’ancienne
intensité de bruit incohérente reste exacte (lignes 2408–2423). Exp2 emploie la
bonne convolution EI et n’utilise jamais le mélange drift \(\eta=0.5\), bruit
\(\eta=1\).

Une seconde différence est explicite : v22 indique encore qu’un score de cible
continue est estimé par Monte-Carlo (lignes 2404–2406 et 2462–2469). Exp2 est un
protocole **exact-score tabulé** : la fonction mathématique visée est le score
marginal exact de chaque modèle, évalué numériquement par inversion spectrale,
interpolation et extrapolation asymptotique. Il n’y a ni réseau ni importance
sampling. « Exact-score » ne signifie donc pas arithmétique exacte ; l’erreur de
table reste une erreur numérique à contrôler.

## Spécification mathématique indépendante

### Cible

La cible est la Student centrée standard à quatre degrés de liberté,

\[
  \pi(x)=\frac{3}{8}\left(1+\frac{x^2}{4}\right)^{-5/2},
  \qquad
  \varphi_\pi(u)=2u^2K_2(2|u|),
\]

avec \(\varphi_\pi(0)=1\) par continuité et
\(\operatorname{Var}(X_0)=2\). Elle est centrée et son indice de queue est
\(\nu=4\). Pour \(\alpha=1.5\), elle satisfait donc
\(\nu>1+\alpha\), le régime des résultats de mélange stable de v22, sous les
autres hypothèses de ces résultats. La loi Student-\(t_4\) apparaît déjà dans
les expériences de v22 (lignes 636–641 et 2615–2626), mais le choix précis
scipy.stats.t(df=4, loc=0, scale=1) est un choix figé d’Exp2.

Implémentation : **experiments/exp2/theory.py:46** et **:54** ; configuration :
section target des trois TOML.

### Forward stable

La convention stable du manuscrit est

\[
  \mathbb E e^{iuL_t^\alpha}=e^{-t|u|^\alpha},\qquad
  L_h^\alpha\overset{\mathcal L}=h^{1/\alpha}L_1^\alpha
\]

(v22, lignes 133–143). Avec \(\beta(t)\equiv\beta_0=1\), le processus forward
est

\[
  \mathrm dX_t=-\frac{\beta_0}{\alpha}X_t\,\mathrm dt
  +\beta_0^{1/\alpha}\,\mathrm dL_t^\alpha,
  \qquad \alpha=1.5,
\]

et

\[
  a_\alpha(t)=e^{-\beta_0t/\alpha},\qquad
  \gamma_\alpha(t)^\alpha=1-e^{-\beta_0t},\qquad
  X_t\overset{\mathcal L}=a_\alpha(t)X_0+\gamma_\alpha(t)Z_\alpha.
\]

Sa fonction caractéristique exacte est

\[
  \varphi_t^{(\alpha)}(u)
  =\varphi_\pi(a_\alpha(t)u)
   \exp\{-\gamma_\alpha(t)^\alpha|u|^\alpha\}.
\]

Ces formules sont celles de v22, équations spsde et scales, lignes 144–153.
Elles sont implémentées dans **theory.py:73**, **:89** et **:124**.

### Backward stable et sens du temps

On pose \(t=T-\tau\), \(\bar\beta(\tau)=\beta(T-\tau)\), avec
\(\tau\in[0,T-\varepsilon]\). La famille de Popov à marginales concordantes
est, dans la notation de v22,

\[
 \mathrm dY_\tau=
 \left[\frac{\bar\beta(\tau)}{\alpha}Y_\tau
 +(1+\eta)\bar\beta(\tau)S_{T-\tau}^{(\alpha)}(Y_\tau)\right]\mathrm d\tau
 +(\eta\bar\beta(\tau))^{1/\alpha}\mathrm d\bar L_\tau^\alpha.
\]

Avec l’initialisation exacte \(Y_0\sim p_T\), sa marginale est \(p_{T-\tau}\)
(v22, Proposition prop:family, lignes 167–183). Exp2 fixe
\(\eta=0.5=\alpha-1\). Cette dynamique additive partage les marginales du vrai
renversement, mais n’est pas en général sa loi de trajectoire (v22, lignes
162–187).

Le score fractionnaire est celui des lignes 155–159. La formule de Tweedie
(v22, Proposition prop:tweedie, lignes 203–210) donne en dimension un

\[
 S_t^{(\alpha)}(x)
 =-\frac{x-a_\alpha(t)m_t(x)}{\alpha\gamma_\alpha(t)^\alpha},
 \qquad m_t(x)=\mathbb E[X_0\mid X_t=x].
\]

### Forward et backward VP, traités séparément

Le baseline VP n’utilise ni le score fractionnaire stable ni la valeur
\(\eta=0.5\). Il suit sa propre diffusion brownienne

\[
 \mathrm dX_t=-\frac{\beta_0}{2}X_t\,\mathrm dt
 +\sqrt{\beta_0}\,\mathrm dW_t,
 \quad
 a_2(t)=e^{-\beta_0t/2},\quad
 \gamma_2(t)^2=1-e^{-\beta_0t},
\]

de loi stationnaire \(\mathcal N(0,1)\), puis sa propre SDE reverse

\[
 \mathrm dY_\tau=
 \left[\frac{\bar\beta(\tau)}{2}Y_\tau
 +\bar\beta(\tau)s_{T-\tau}(Y_\tau)\right]\mathrm d\tau
 +\sqrt{\bar\beta(\tau)}\,\mathrm d\bar W_\tau,
\]

où \(s_t=\partial_x\log p_t^{(2)}\). Ces expressions sont données dans v22,
lignes 1817–1833. On a

\[
 \varphi_t^{(2)}(u)=\varphi_\pi(a_2(t)u)
 \exp\{-\tfrac12\gamma_2(t)^2u^2\},\qquad
 s_t(x)=-\frac{x-a_2(t)m_t^{(2)}(x)}{\gamma_2(t)^2}.
\]

Les références exactes sont donc propres au modèle :
\(\mathrm S\alpha\mathrm S(1)\) pour stable et \(\mathcal N(0,1)\) pour VP ;
les marginales \(p_T\), les scores et les échantillons \(p_t\) sont également
propres à chaque forward. La comparaison ne doit pas être décrite comme deux
intégrateurs appliqués à un score commun.

### Score exact tabulé

À partir des fonctions caractéristiques précédentes, le code reconstruit la
densité marginale et le numérateur du score par FFT. Le multiplicateur est
\(-iu|u|^{\alpha-2}\) côté stable et \(-iu\) côté VP. Le quotient donne le score
sur une grille spatiale positive, l’imparité étant imposée exactement
(**score_tables.py:342–411**). Les tables sont en float64, sur des temps
géométriques de \(\varepsilon\) à \(2\) (**score_tables.py:147–181**).

Pour éviter l’instabilité temporelle près de \(\varepsilon\), l’interpolation
porte sur le résidu régulier

\[
 R_t(x)=x+\alpha\gamma_\alpha(t)^\alpha S_t^{(\alpha)}(x)
       =a_\alpha(t)m_t(x)
\]

côté stable, et sur
\(R_t^{(2)}(x)=x+\gamma_2(t)^2s_t(x)=a_2(t)m_t^{(2)}(x)\) côté VP
(**score_tables.py:477** et **:554**). Une transition \(C^1\) relie la FFT à une
expansion asymptotique spécifique au modèle, et des points hors grille sont
comparés à une quadrature de Fourier indépendante (**score_tables.py:414** et
**:703**).

Cette construction élimine l’erreur d’apprentissage, mais pas l’erreur de
tabulation, d’interpolation ou d’extrapolation. La composante \(D\) définie
ci-dessous les contient avec l’erreur EI. Chaque tâche produit désormais un
bras exact_p_T haute résolution apparié au bras principal par la même
initialisation et les mêmes primitives. Les distances `score_sensitivity_wp`
sont enregistrées et `d_controlled` n'est vrai que si elles restent sous la
fraction préspécifiée de \(F\). Le pilote courant contrôle 1 280/1 280 points.

### Exponential integrator

La reformulation dissipative de v22 (équation dissip_reform, lignes 308–321,
et équations app_reform–app_ei, lignes 1186–1207) écrit la dynamique stable
sous la forme \(-\lambda x+F(x,t)\), avec

\[
 \lambda_\alpha=\frac{\eta\beta_0}{\alpha},\qquad
 F_\alpha(x,t)=\frac{(1+\eta)\beta_0}{\alpha}
                  \bigl(x+\alpha S_t^{(\alpha)}(x)\bigr).
\]

Sur la grille backward uniforme
\(\tau_k=kh\), \(h=(T-\varepsilon)/N\), \(t_k=T-kh\), le score est évalué au
point gauche \(t_k\), avant la mise à jour, et

\[
 Y_{k+1}=e^{-\lambda_\alpha h}Y_k
 +\frac{1-e^{-\lambda_\alpha h}}{\lambda_\alpha}F_\alpha(Y_k,t_k)
 +(1-e^{-\eta\beta_0h})^{1/\alpha}\xi_k,
 \quad \xi_k\sim\mathrm S\alpha\mathrm S(1).
\]

Avec \((\alpha,\eta,\beta_0)=(1.5,0.5,1)\), l’intensité est exactement
\(\sigma_h^\alpha=1-e^{-0.5h}\), et non \(1-e^{-h}\), ni
\(\eta h\). La dérivation est explicitée dans v22, lignes 2471–2513.
L’implémentation se trouve dans **src/levy_experiments/integrators.py:20** et
**experiments/exp2/simulation.py:111–254**.

Le VP possède son EI distinct,

\[
 \lambda_2=\frac{\beta_0}{2},\qquad
 F_2(x,t)=\beta_0(x+s_t(x)),\qquad
 Y_{k+1}=e^{-\lambda_2h}Y_k
 +\frac{1-e^{-\lambda_2h}}{\lambda_2}F_2(Y_k,t_k)
 +\sqrt{1-e^{-\beta_0h}}\,\zeta_k,
\]

avec \(\zeta_k\sim\mathcal N(0,1)\). Chaque trajectoire utilise exactement
\(N\) évaluations du score. Les checkpoints de fraction \(r\) sont enregistrés
après l’étape \(rN\), au temps forward
\(t=T-r(T-\varepsilon)\) (**simulation.py:47**, **:83** et **:193–249**).

## Matrice de traçabilité

| Élément | Autorité courante | Implémentation Exp2 | Statut et conséquence |
|---|---|---|---|
| Convention stable | Lignes 133–143 | theory.py:89–153, générateur partagé | Conforme : CF \(e^{-|u|^\alpha}\), échelle \(h^{1/\alpha}\). |
| Forward stable | Éq. spsde, scales, lignes 144–153 | theory.py:73, :89, :124 | Conforme, \(\beta_0=1,\alpha=1.5\). |
| Score fractionnaire | Éq. score_def, tweedie, lignes 155–159 et 203–210 | score_tables.py:342–411, :477–699 | Score mathématique exact, évaluation tabulée numérique ; pas de réseau. |
| Temps backward | Proposition prop:family, lignes 169–183 | simulation.py:83–92, :195 | Conforme : \(t=T-\tau\), arrêt à \(t=\varepsilon=0.05\). |
| Drift stable | Éq. backward_drift, lignes 170–176 | integrators.py:59, simulation.py:203–208 | Conforme pour \(\eta=0.5\). |
| Bruit stable EI | Éq. app_ei_noise, lignes 2492–2511 | integrators.py:20–44, simulation.py:141–151 | Conforme : \(\sigma_h^\alpha=1-e^{-\eta\beta_0h}\). |
| Forward/reverse VP | Lignes 1817–1833 | theory.py:73–153, simulation.py:34–44, :221–243 | Baseline séparé, score ordinaire propre et référence normale propre. |
| Initialisation exacte/référence | Décomposition lignes 286–305 ; théorèmes lignes 443–481 | simulation.py:155–191, config.py:136–143 | Trois bras sont présents : exact-pT main et hires sont appariés ; le bras stationnaire reste distinct. |
| \(p<\alpha\) | Lignes 331–347, 453–456, 489–495 et 1229–1251 | config.py:553–710, analysis.py:393–468 | Ordres figés \(p=1,1.25<1.5\) ; aucun \(W_2\) stable. |
| Raffinement EI | Proposition prop:order, lignes 483–503 | plan design, analysis.py:655–722 | Seulement à \((\varepsilon,T)=(0.05,2)\) fixé ; pas de limite jointe. |
| Mélange avec \(T\) | Cor. cor:mixing et Th. gauss_init, lignes 443–481 | analysis.py:724–838 | Ajustement descriptif de \(I\), par seed et par modèle ; aucune comparaison de pente stable–VP. |
| Ancienne erreur backward | Remarque app:exp:status, lignes 2408–2423 | EI corrigé ci-dessus | Exp2 n’emploie pas l’ancien bruit \(\eta=1\) avec drift \(\eta=0.5\). |

## Plan expérimental effectivement figé

Paramètres communs : dimension \(1\), Student-\(t_4\) standard,
\(\beta_0=1\), \(\varepsilon=0.05\), stable
\((\alpha,\eta)=(1.5,0.5)\), précision float64, trois bras propagés
(exact_p_T main, stationary_reference main, exact_p_T hires apparié), et trois
références exactes indépendantes A/B/C.

| Niveau | Seeds | Particules | Batch | Sweep \((T,N)\) | Raffinement à \(T=2\) | Checkpoints backward | Tâches uniques |
|---|---:|---:|---:|---|---|---|---:|
| smoke | 0–1 | 2 048 | 1 024 | (0.25,4), (2,8) | \(N=8,16\) | 0, 1/2, 1 | 12 |
| pilot | 200–207 | 8 192 | 8 192 | (0.1,4), (0.15,8), (0.25,16), (0.5,36), (1,76), (2,156) | \(N=156,312,624\) | 0, 1/4, 1/2, 3/4, 1 | 128 |
| final | 1000–1011 | 16 384 | 8 192 | même sweep | \(N=156,312,624\) | 0, 1/4, 1/2, 3/4, 1 | 192 |

Les points \((T,N)\) présents à la fois dans le sweep et le raffinement ne sont
pas dupliqués. Une tâche est l’unique quadruplet (model,T,N,seed) et contient
les trois bras propagés ainsi que A/B/C.

### Tables de score propres aux modèles

| Niveau | Temps | Espace | Stable : FFT, \(L\), \(x_{\max}\), raccord | VP : FFT, \(L\), \(x_{\max}\), raccord |
|---|---:|---:|---|---|
| smoke | 17 | 513 | 131 072 ; 2 048 ; 192 ; 128–192 | 32 768 ; 256 ; 32 ; 12–16 |
| pilot | 129 | 2 049 | 262 144 ; 4 096 ; 256 ; 192–256 | 131 072 ; 512 ; 64 ; 12–16 |
| final | 257 | 4 097 | 524 288 ; 8 192 ; 384 ; 256–384 | 262 144 ; 1 024 ; 96 ; 12–16 |

Ici \(L\) est la demi-largeur du domaine spatial de la FFT. Les domaines stable
ont été étendus séparément : réutiliser le raccord VP pour la loi stable ferait
intervenir l’asymptotique trop tôt.

Chaque niveau possède en outre une table indépendante haute résolution avec
respectivement `(time,space)=(33,1025),(257,4097),(513,8193)` pour
smoke/pilot/final et deux fois le nombre de points FFT du modèle principal. Le
bras hires partage exactement l'initialisation et les primitives aléatoires du
bras exact_p_T principal ; sa distance au bras principal est la sensibilité de
table utilisée par la porte fail-closed.

## Observables \(D,I,E,F\)

À un checkpoint de temps forward \(t\), notons
\(\widehat\mu_{t,h}^{T}\) la loi empirique EI initialisée par \(p_T\),
\(\widehat\mu_{t,h}^{\infty}\) celle initialisée par la référence stationnaire,
et \(\widehat p_t^A,\widehat p_t^B,\widehat p_t^C\) trois échantillons exacts
indépendants de la marginale forward propre au modèle. Pour
\(p\in\{1,1.25\}\), l’analyse calcule

\[
\begin{aligned}
 D_p(t;T,h)&=W_p(\widehat\mu_{t,h}^{T},\widehat p_t^C),\\
 I_p(t;T,h)&=W_p(\widehat\mu_{t,h}^{\infty},\widehat\mu_{t,h}^{T}),\\
 E_p(t;T,h)&=W_p(\widehat\mu_{t,h}^{\infty},\widehat p_t^C),\\
 F_p(t)&=W_p(\widehat p_t^A,\widehat p_t^B).
\end{aligned}
\]

- \(D\) mesure empiriquement la combinaison erreur EI + erreur de table +
  fluctuation d’échantillonnage, sans erreur d’initialisation asymptotique.
- \(I\) est un **proxy à pas fini** de l’erreur d’initialisation entre les deux
  lois empiriques numériques indépendantes et inclut donc leur résolution
  d’échantillonnage ; ce n’est ni un écart trajectoriel synchrone ni exactement
  le terme continu \(\mathcal E_1\) de v22.
- \(E\) est l’écart empirique total du sampler initialisé à la référence vers
  \(p_t\), avec sa fluctuation d’échantillonnage.
- \(F\) est le plancher empirique exact–exact ; il n’est jamais soustrait.
  A/B lui sont réservés, tandis que C reste indépendant des comparaisons D/E.

Le transport empirique unidimensionnel est exact pour les mesures empiriques de
même taille,

\[
 W_p(\hat\mu_n,\hat\nu_n)
 =\left(\frac1n\sum_{i=1}^n|X_{(i)}-Y_{(i)}|^p\right)^{1/p}.
\]

La triangulaire \(E\le I+D\) est vérifiée numériquement. Elle ne justifie pas
de remplacer \(E\) par \(I+D\), ni de soustraire \(F\).

## ECF, tailles d’échantillon et agrégation

L’ECF est évaluée sur 81 fréquences uniformes de \([-5,5]\), par blocs de
4 096 observations, avec poids \(w(u)=e^{-u^2/2}\). Pour chaque loi numérique
et pour les références A/B/C, l’analyse produit

\[
 \left(\frac{\sum_jw_j|\widehat\varphi(u_j)-\varphi_t(u_j)|^2}
 {\sum_jw_j}\right)^{1/2}
\]

et l’erreur maximale des composantes réelle/imaginaire. Le rayon de Hoeffding
simultané enregistré est

\[
 r_{n,m}=\sqrt{\frac{2\log(4m/0.05)}{n}},\qquad m=81.
\]

Les métriques sont recalculées sur les fractions imbriquées 1/4, 1/2 et sur
l’échantillon complet ; quatre blocs disjoints sont également utilisés pour
un diagnostic intra-seed. Les blocs sont d’abord résumés dans chaque seed, puis
les seeds indépendantes sont agrégées par médiane et quantiles 16 %/84 %.
Les fractions sont figées dans les TOML ; le nombre de blocs quatre est figé
dans **aggregate.py:385–393**, donc par l’empreinte du code plutôt que par le TOML.

## Règles d’interprétation et de raffinement

1. Le raffinement en \(h\) utilise uniquement \(T=2\) et
   \(\varepsilon=0.05\) fixes. Il ne teste pas une limite jointe
   \(T\to\infty\), \(\varepsilon\downarrow0\), \(h\downarrow0\).
2. Une régression \(\log D\) sur \(\log h\) exige au moins trois pas distincts,
   \(D>2F\), et un contrôle de table (controlled=true). En l’absence de ce
   dernier, aucune pente n’est publiée. La porte finale exige en plus une
   couverture séparée suffisante pour stable/VP et \(p=1,1.25\).
3. Le sweep en \(T\) sélectionne le plus petit \(h\) disponible à chaque
   horizon. Une régression descriptive \(\log I\) sur \(T\) exige au moins trois
   horizons, \(I>2F\), et \(D\le\max(2F,0.25I)\).
4. Les ajustements sont effectués seed par seed. \(E\) n’est pas ajusté car il
   mélange initialisation et numérique. Aucune pente stable–VP n’est testée :
   v22 conclut précisément qu’aucune séparation intrinsèque claire ne découle
   du seul mélange en \(W_p\) (lignes 468–481).
5. Le smoke ne possède que deux horizons et deux pas de raffinement. Il valide
   le flux logiciel, les identités, les formats et les diagnostics ; il ne peut
   soutenir ni une loi de convergence en \(h\), ni un taux exponentiel en \(T\).

## Conclusions permises, sous conditions

| Question | Observable | Condition minimale | Formulation permise |
|---|---|---|---|
| L’EI approche-t-il la marginale exacte à \(p_T\) exact ? | \(D\), ECF | table contrôlée, \(D>F\), raffinement pilot/final | « convergence numérique compatible avec le raffinement observé », pas preuve d’ordre sans fit admissible |
| L’initialisation stationnaire devient-elle moins influente quand \(T\) croît ? | \(I\) terminal | \(I>2F\), \(D\) non confondant, au moins 3 horizons | « décroissance descriptive de l’effet d’initialisation dans ce protocole discret » |
| L’erreur totale est-elle résolue ? | \(E\) contre \(F\), ECF | taille/seed suffisantes et stabilité au sous-échantillonnage | « écart empirique résolu/non résolu au-dessus du plancher » |
| Stable et VP mélangent-ils tous deux ? | séries propres à chaque modèle | critères précédents satisfaits séparément | deux constats intra-modèle ; aucune supériorité de taux entre modèles |
| Le théorème continu est-il validé ? | aucun observable fini ne suffit | — | au plus une illustration numérique exact-score, temps fini, pas fini |

## Statut d’exécution et commandes

Le pilote courant a terminé 128/128 tâches et sa porte de publication a passé.
Le run final n'a pas été exécuté. Le reçu complet est documenté dans
`reports/final_gate_validation.md`. Commandes smoke de reproduction :

    .venv\Scripts\python.exe -m experiments.exp2.run ^
      --config configs/smoke/exp2.toml --device cpu --precision float64 ^
      --output-dir outputs
    .venv\Scripts\python.exe -m experiments.exp2.aggregate ^
      --config configs/smoke/exp2.toml --device cpu --precision float64 ^
      --output-dir outputs

Le dry-run final peut être inspecté sans autorisation. Toute exécution finale
exige `--allow-publication-scale` et le reçu pilote complet sous
`--pilot-output-dir`. Les nombres du pilote sont des diagnostics de sélection
du protocole ; ils ne doivent pas être cités comme résultats finaux.

## Modifications à prévoir dans v22 si Exp2 est retenue

1. Présenter Exp2 comme un protocole exact-score tabulé, et retirer pour cette
   expérience la phrase « continuous-target score is evaluated by Monte Carlo ».
2. Ne pas conserver la description de couplage synchrone des lignes 2517–2534
   si Exp2 remplace le panneau dynamique (c) ; définir \(D,I,E,F\) à la place.
3. Ajouter les deux forward, scores, références et EI comme objets séparés.
4. Écrire \(p<\alpha\) au voisinage immédiat des résultats stable et limiter les
   figures à \(W_1,W_{1.25}\).
5. Qualifier \(I\) de proxy d’initialisation à pas fini, \(D\) d’erreur
   numérique incluant la table, \(E\) d’erreur totale et \(F\) de plancher non
   soustrait.
6. Préciser que le raffinement se fait à \((\varepsilon,T)\) fixé et que les
   fits sont conditionnels aux règles de résolution ci-dessus.
7. Ne publier aucune statistique du pilote comme résultat final et ne présenter
   aucune simulation finie comme preuve d'un théorème.
