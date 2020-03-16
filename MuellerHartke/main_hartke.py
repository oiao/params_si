import os
from os.path import join as opj
import numpy as np

from scm.params import *
from scm.plams import Settings

INDIR = 'input_params'

### STEP 1: Convert old reax formats to params
if not os.path.exists(INDIR):
    os.makedirs(INDIR)

def add_grads(path, dataset):
    factor = 1. #Units.convert(1.0, 'au', 'kcal/mol') * Units.convert(1.0, 'bohr^-1', 'angstrom^-1')
    for i in os.listdir(path):
        if i.endswith('gradient'):
            name = i.rstrip('.gradient')
            grads = factor * np.loadtxt(opj(path, i), skiprows=1, usecols=(1,2,3))

            for id,at in enumerate(grads):
                for xyz,value in enumerate(at):
                    dataset.add_entry(f'force("{name}", {id}, {xyz})', weight=0.01, reference=-value)


print('Converting geo files ...')
s = Settings()
s.input.ams.properties.gradients = True # We need to specifically request gradients with AMS
jc1 = geo_to_params('SI/optInput/geo', normal_run='GO', settings=s)
jc2 = geo_to_params('SI/valSet/geo',   normal_run='GO', settings=s)
print('The following jobIDs are in *both* the training and validation sets:')
print("\n".join([i for i in jc1.keys() if i in jc2])+'\n')
jc  = jc1 + jc2
jc.store(opj(INDIR, 'jobcollection.yml'))

print('Converting trainset.in files ...')
train_set = trainset_to_params('SI/optInput/trainset.in')
add_grads('SI/optInput/grads', train_set)
val_set  =  trainset_to_params('SI/valSet/trainset.in')
train_set.store(opj(INDIR, 'trainingset.yml'))
val_set.store(  opj(INDIR, 'validationset.yml'))

print(f"Params converted sets stored to .{os.sep+INDIR}")


### STEP 2: Calculate the fx value of x0 and x*
x0   = ReaxParams('SI/ffield_mattson_modified')
xopt = ReaxParams('SI/ffield_best')

print()
for i in [x0, xopt]: # Do the same twice
    if i == x0: print('Running x0 ...')
    else:       print('Running x_opt ...')
    engine = i.get_engine()
    r  = jc.run(engine.settings)
    fx = train_set.evaluate({i.name : i for i in r}, 'sse')
    print(f'Training Set   f(x) = {fx:.3e}')
    fx = val_set.evaluate({i.name : i for i in r}, 'sse')
    print(f'Validation Set f(x) = {fx:.3e}\n')
print('Published training set value is 16271 (doi.org/10.1021/acs.jctc.9b00769)')



### STEP 3: Set up an optimization
x = xopt # Select which ffield to optimize here

# Use this mask to select the .active subset of parameters for the optimization
is_active_mask = [x0[i] != xopt[i] for i in x0.names] # All values that differ between two ffields will be optimized
x.is_active = is_active_mask

print(f"\nWill optimize (len.xopt
o = CMAOptimizer(popsize=10, sigma=0.25)
callbacks = [Logger(plot=True), Timeout(60*60), TimePerEval(5), EarlyStopping(1000)]
optimization = Optimization(jc, [train_set, val_set], xopt, o, callbacks=callbacks)

optimization.summary()
optimization.optimize()

print('*** All done! ***')
