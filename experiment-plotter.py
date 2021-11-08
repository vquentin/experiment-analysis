# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

import logging
from experiments.loader import get_experiment

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s')

SEM_UNIFORMITY = 'uniformity-SEM-CS'
SEM_UNIFORMITY_NORMALIZED = 'uniformity-SEM-CS-normalize'
U_VS_T = 'u-vs-t'
MOSS = 'MOSS'

if __name__ == '__main__':
    Ag_high_cd = ('WF248', 'WF164', 'WF185')
    metal_masks_Au_Ag = ('WF74', 'WF148',)
    Au_different_Cr_thickness = ('WF74', 'WF67', 'WF102')

    Ag_SiN_no_adhesion = ('WF148', 'WF42')

    time_effect_on_uniformity_with_SiN = ('WF56', 'WF66')
    metal_masks = ('WF56', 'WF74')
    #metal_masks_Au_Ag = ('WF56', 'WF74', 'WF148')
    metal_masks_Ag_pretreatment = ('WF148', 'WF149', 'WF146', 'WF147')
    metal_masks_Au_Cr = ('WF74', 'WF67', 'WF102')
    metal_masks_dielectric = ('WF42','WF74', 'WF43', 'WF148')
    metal_masks_dielectric_Al = ('WF56', 'WF162', 'WF47')
    metal_masks_SiN = ('WF42','WF43','WF47', 'WF48','WF248')
    metal_masks_implant = ('WF175',)
    resistivity = ('ADB1', 'WF56')
    metal_masks_Al_time = ('WF245', 'WF246')
    justification_Al_adhesion = ('WF56', 'WF148', 'WF162', 'WF47', 'WF42', 'WF175')
    Al_vs_antiporous = ('WF162', 'WF47', 'WF175')
    thick_porous_attempts = ('WF48', 'WF245', 'WF248')
    best_sample = ('WF248',)
    Al_thickness_dependence = ('WF48', 'WF245', 'WF246', 'WF247')
    MOSS_experiments = ('WF54', 'WF158', 'WF125')

    legend = 'mask.dielectric.material'
    #legend = 'mask.material+mask.dielectric.material'
    #legend = 'mask.material+mask.adhesionLayer'
    #legend = 'mask.material'
    #legend = 'porosification.time'
    #legend = 'mask.adhesionLayer'
    #legend = 'silicon.resistivity.value'
    #legend = 'mask.pretreatment'
    #legend = 'mask.adhesion.thickness'

    #experiment = get_experiment(MOSS, *MOSS_experiments)
    experiment = get_experiment(U_VS_T, *Ag_high_cd)
    #experiment = get_experiment(SEM_UNIFORMITY_NORMALIZED, *Ag_high_cd)
    #experiment = get_experiment(SEM_UNIFORMITY, *newSample)

    experiment.run()
    experiment.save()
    experiment.plot(legend)
