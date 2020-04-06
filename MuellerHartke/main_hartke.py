#!/usr/bin/env python
import os, sys
from os.path import join as opj
import numpy as np

from scm.params import *

from scm.params.common.helpers import _cleanup_example_folder
_cleanup_example_folder(['opt*'])

def add_grads(path, dataset):
    factor = 1. #Units.convert(1.0, 'au', 'kcal/mol') * Units.convert(1.0, 'bohr^-1', 'angstrom^-1')
    for i in os.listdir(path):
        if i.endswith('gradient'):
            name = i.rstrip('.gradient')
            grads = factor * np.loadtxt(opj(path, i), skiprows=1, usecols=(1,2,3))

            for id,at in enumerate(grads):
                for xyz,value in enumerate(at):
                    dataset.add_entry(f'force("{name}", {id}, {xyz})', weight=0.01, reference=-value)

def get_constraints(x):
    from itertools import combinations
    pre_atoms = ['C:', 'H:', 'O:', 'S:']
    pre_offd = [f"{'.'.join(i)}:" for i in combinations(['C', 'H', 'O', 'S'], 2)]
    const = []
    for pre in pre_atoms:
        sigma = f"{pre}r_0^sigma;;2;;Sigma bond covalent radius"
        pi    = f"{pre}r_0^pi;;2;;Pi bond covalent radius"
        pi2   = f"{pre}r_0^pi;;2;;Double pi bond covalent radius"
        const += [x[sigma] >= x[pi], x[pi] >= x[pi2]]
    for pre in pre_offd:
        sigma = f"{pre}r_0^sigma;;2;;Sigma bond length"
        pi    = f"{pre}r_0^pi;;2;;Pi bond length"
        pi2   = f"{pre}r_0^pipi;;2;;PiPi bond length"
        const += [x[sigma] >= x[pi], x[pi] >= x[pi2]]
    return const



INDIR = 'input_params'
if not os.path.exists(INDIR):
    os.makedirs(INDIR)



### STEP 1: Convert old reax formats to params
print('Converting geo files ...')
jc1 = geo_to_params('SI/optInput/geo', normal_run_settings='SI/control')
jc2 = geo_to_params('SI/valSet/geo',   normal_run_settings='SI/control')
print('The following jobIDs are in *both* the training and validation sets:')
print("\n".join([i for i in jc1.keys() if i in jc2])+'\n')

jc  = jc1 + jc2
for e in jc.values():
    e.metadata['Source'] = 'https://doi.org/10.1021/acs.jctc.6b00461'
    e.settings.input.ams.properties.gradients = True # We need to specifically request gradients with AMS
jc.store(opj(INDIR, 'jobcollection.yml'))

print('Converting trainset.in files ...')
train_set = trainset_to_params('SI/optInput/trainset.in')
add_grads('SI/optInput/grads', train_set)
val_set  =  trainset_to_params('SI/valSet/trainset.in')
for e in train_set+val_set:
    e.metadata['Source'] = 'https://doi.org/10.1021/acs.jctc.6b00461'
train_set.store(opj(INDIR, 'trainingset.yml'))
val_set.store(  opj(INDIR, 'validationset.yml'))

print(f"Params converted sets stored to .{os.sep+INDIR}")



### STEP 2: Calculate the fx value of x0 and x*
x0   = ReaxParams('SI/ffield_mattson_modified', 1.15)
xopt = ReaxParams('SI/ffield_best', 1.15)

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
x = xopt # Select which ffield to optimize here: xopt, x0

# Use this mask to select the .active subset of parameters for the optimization
is_active_mask = [x0[i] != xopt[i] for i in x0.names] # All values that differ between two ffields will be optimized
x.is_active = is_active_mask

# All jobs should run through the pipe
badids = [i for i,e in jc.items() if not e.is_pipeable()]
assert badids == [], 'Some jobs are not pipeable!'


print('\nNow optimizing:\n')
o = CMAOptimizer(popsize=10, sigma=0.25)

callbacks = [Logger(plot=False), Timeout(60*60*2), TimePerEval(10), EarlyStopping(1000)]
constraints = get_constraints(x)
optimization = Optimization(jc, [train_set, val_set], x, o, callbacks=callbacks, constraints=constraints)

optimization.summary()
optimization.optimize()

print('*** All done! ***')
