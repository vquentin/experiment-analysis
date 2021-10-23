# Copyright (c) 2021, Quentin Van Overmeere
# Licensed under MIT License

from experiments.loader import get_experiment
import logging

logging.basicConfig(format='%(asctime)s %(name)s %(levelname)s:%(message)s')

SEM_UNIFORMITY = 'uniformity-SEM-CS'
SEM_UNIFORMITY_NORMALIZED = 'uniformity-SEM-CS-normalize'
U_VS_T = 'u-vs-t'
MOSS = 'MOSS'

if __name__ == '__main__':
    #samples = ('ADB1',)
    #samples = ('UB19',)
    #newSample=('WF42',)
    time_effect = ('WF56', 'WF66')
    metal_masks = ('WF56', 'WF74')
    metal_masks_Au_Ag = ('WF56', 'WF74', 'WF148')
    metal_masks_Ag_pretreatment = ('WF148', 'WF149', 'WF146', 'WF147')
    metal_masks_Au_Cr = ('WF74', 'WF67', 'WF102')
    metal_masks_dielectric = ('WF42','WF74', 'WF43', 'WF148')
    metal_masks_dielectric_Al = ('WF56', 'WF162', 'WF47')

    metal_masks_SiN = ('WF47', 'WF48')

    metal_masks_implant = ('WF175',)

    resistivity = ('ADB1', 'WF56')

    metal_masks_Al_time = ('WF245', 'WF246')

    justification_Al_adhesion = ('WF56', 'WF148', 'WF162', 'WF47', 'WF42', 'WF175')
    Al_vs_antiporous = ('WF162', 'WF47', 'WF175')
    thick_porous_attempts = ('WF48', 'WF245','WF248')
    best_sample = ('WF248',)
    Al_thickness_dependence = ('WF48','WF245', 'WF246', 'WF247')


    MOSS_experiments = ('WF54',)

    #legend='mask.material+mask.dielectric.material'
    #legend='mask.material'
    #legend='porosification.time'
    legend='name'
    #legend = 'mask.adhesionLayer'
    #legend = 'silicon.resistivity.value'
    #legend = 'mask.pretreatment'
    #legend = 'mask.adhesion.thickness'
    #samples = ('examples',)
    #experiment = get_experiment(SEM_UNIFORMITY_NORMALIZED, *metal_masks_Al_time)
    experiment = get_experiment(SEM_UNIFORMITY_NORMALIZED, *thick_porous_attempts)

    #experiment = get_experiment(MOSS, *MOSS_experiments)

    
    #experiment = get_experiment(SEM_UNIFORMITY_NORMALIZED, *metal_masks_implant)
    experiment = get_experiment(U_VS_T, *best_sample)
    #experiment = get_experiment(SEM_UNIFORMITY_NORMALIZED, *Al_thickness_dependence)
    #experiment = get_experiment(SEM_UNIFORMITY_NORMALIZED, *metal_masks_Ag_pretreatment)
    #experiment = get_experiment(SEM_UNIFORMITY_NORMALIZED, *metal_masks_Au_Cr)
    #experiment = get_experiment(SEM_UNIFORMITY_NORMALIZED, *resistivity)
    #experiment = get_experiment(SEM_UNIFORMITY_NORMALIZED, *metal_masks_dielectric)
    #experiment = get_experiment(SEM_UNIFORMITY_NORMALIZED, *metal_masks_dielectric_Al)
    #experiment = get_experiment(SEM_UNIFORMITY, *newSample)
    experiment.run()
    #experiment.save()
    #experiment.plot(legend)
    experiment.plot(legend)
   

    #experiment2 = get_experiment(SEM_UNIFORMITY, *metal_masks)
    #experiment2.run()
    #experiment2.plot(legend)
    #experiment2.save()
