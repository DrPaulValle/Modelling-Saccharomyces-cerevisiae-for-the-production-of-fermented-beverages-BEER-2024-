[![Open in MATLAB Online](https://www.mathworks.com/images/responsive/global/open-in-matlab-online.svg)](https://matlab.mathworks.com/open/github/v1?repo=DrPaulValle/Modelling-Saccharomyces-cerevisiae-for-the-production-of-fermented-beverages-BEER-2024-)

# Modelling Saccharomyces cerevisiae for the production of fermented beverages

# Authors
Paul A. Valle, Yolocuauhtli Salazar, Luis N. Coria, Nicolas O. Soto-Cruz, Jesus B. Paez-Lerma
Postgraduate Program in Engineering Sciences, BioMath Research Group, Tecnologico Nacional de Mexico/IT Tijuana, Blvd. Alberto Limon Padilla s/n, Tijuana 22454, Mexico
Postgraduate Program in Engineering, Tecnologico Nacional de Mexico/IT Durango, Blvd. Felipe Pescador 1830 Ote., Durango 34080, Mexico
Departament Chemical and Biochemical Engineering, Tecnologico Nacional de Mexico/IT Durango, Blvd. Felipe Pescador 1830 Ote., Durango 34080, Mexico

# Abstract
Mezcal is a traditional distilled spirit beverage from the state of Durango, Mexico, which is obtained by the fermentation of Agave duranguensis sugars. In this work, we formulate a mechanistic model consisting of three first-order Ordinary Differential Equations (ODEs) to describe ethanol production for mezcal by the Saccharomyces cerevisiae yeast using glucose and fructose as substrates. These equations were fitted to three different sets of experimental data, representing the concentrations of glucose [x(t)], fructose [y(t)], and ethanol [z(t)] in g/L per hour. A nonlinear regression algorithm was designed in MATLAB to fit the model to each set of experimental data where values of parameters were estimated with a 95% confidence interval. The goodness of fit of the model was evaluated both quantitatively and qualitatively by means of the coefficient of determination (R-squared), the Akaike Information Criterion (AIC), and numerical simulations that illustrate data approximation and prediction in the short-term. Furthermore, conditions for the existence of a localizing domain in the nonnegative octant as well as sufficient conditions for asymptotic stability of the unique biologically feasible equilibrium point of the system were derived by means of the Localization of Compact Invariant Sets method and Lyapunov's Stability theory for nonlinear autonomous systems.

# Keywords
Alcoholic fermentation; Asymptotic stability; Biostatistcs; In silico experimentation; Mechanistic modelling; Ordinary Differential Equations.

# Experimental data
The S. cerevisiae strain ITD-00185 was obtained from the yeast collection of the Microbial Biotechnology Laboratory at the Durango Institute of Technology. Fermentation kinetics were conducted in triplicate using Agave duranguensis juice with an initial sugar concentration of 129±4.7 g/L (12% glucose and 88% fructose) and a C/N ratio of 73 g C/g N as substrate. Fermentation kinetics were carried out in glass tubes (20×150 mm), each containing 15 mL of agave juice inoculated with 1×10^8 cells/mL. These tubes were then incubated at 28°C for 72 hours. Samples were analyzed every 8 hours using high-performance liquid chromatography (HPLC) to determine the concentrations of glucose, fructose, and ethanol.

# Acknowledgements
This research was fulfilled within the TecNM projects “Estrategias in silico integradas con biomatemáticas y sistemas dinámicos no lineales para el modelizado, análisis y control de sistemas biológicos [19377.24-P]”, “Modelizado de sistemas no lineales para procesos de fermentación basados en la dinámica de crecimiento de microorganismos: parte 2 [19805.24-P]”, the academic research group Project “Sistemas dinámicos no lineales” ITIJ-CA-10, and the RICCA “Red Internacional de Control y Cómputo Aplicados”.

# References
[1] Walker, Graeme M., and Graham G. Stewart. "Saccharomyces cerevisiae in the production of fermented beverages." Beverages 2.4 (2016): 30.

[2] Guerrero, Martha E. Nuñez, et al. "Physiological characterization of two native yeasts in pure and mixed culture using fermentations of agave juice." Ciencia e investigación agraria: revista latinoamericana de ciencias de la agricultura 46.1 (2019): 1-11.

[3] Soto‐Cruz, Oscar, Ernesto Favela‐Torres, and Gerardo Saucedo‐Castañeda. "Modeling of growth, lactate consumption, and volatile fatty acid production by Megasphaera elsdenii cultivated in minimal and complex media." Biotechnology progress 18.2 (2002): 193-200.

[4] Garnier, Alain, and Bruno Gaillet. "Analytical solution of Luedeking–Piret equation for a batch fermentation obeying Monod growth kinetics." Biotechnology and Bioengineering 112.12 (2015): 2468-2474.

[5] Wolfenden, Richard, and Yang Yuan. "Rates of spontaneous cleavage of glucose, fructose, sucrose, and trehalose in water, and the catalytic proficiencies of invertase and trehalas." Journal of the American Chemical Society 130.24 (2008): 7548-7549.

[6] Valle, Paul A., et al. "CAR-T cell therapy for the treatment of ALL: eradication conditions and in silico experimentation." Hemato 2.3 (2021): 441-462.

[7] Salazar, Yolocuauhtli, et al. "Mechanistic modelling of biomass growth, glucose consumption and ethanol production by Kluyveromyces marxianus in batch fermentation." Entropy 25.3 (2023): 497.

[8] Krishchenko, Alexander P., and Konstantin E. Starkov. "Localization of compact invariant sets of the Lorenz system." Physics Letters A 353.5 (2006): 383-388.

[9] Garﬁnkel, Alan, Jane Shevtsov, and Yina Guo. Modeling life: the mathematics of biological systems. Springer International Publishing AG, 2017.

[10] Khalil, Hassan K. Nonlinear systems, 3rd ed. Prentice-Hall, USA, 2002.

[11] Motulsky, Harvey. Intuitive biostatistics: a nonmathematical guide to statistical thinking. Oxford University Press, USA, 2014.
