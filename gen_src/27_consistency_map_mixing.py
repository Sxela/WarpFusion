#@title Consistency map mixing
#@markdown You can mix consistency map layers separately\
#@markdown missed_consistency_weight - masks pixels that have missed their expected position in the next frame \
#@markdown overshoot_consistency_weight - masks pixels warped from outside the frame\
#@markdown edges_consistency_weight - masks moving objects' edges\
#@markdown The default values to simulate previous versions' behavior are 1,1,1

missed_consistency_weight = 1 #@param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
overshoot_consistency_weight = 1 #@param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}
edges_consistency_weight = 1 #@param {'type':'slider', 'min':'0', 'max':'1', 'step':'0.05'}