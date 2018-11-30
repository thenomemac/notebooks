# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.2'
#       jupytext_version: 0.8.6
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# As shown in this github issue: https://github.com/scikit-learn/scikit-learn/issues/5975
#
# I've had many arguements that for poisson regression Counts/Exposures is equivalent to handling the offsets in the optimization loop explicitly. This isn't true as shown later on in this notebook.
#
# The notebook also shows how to do poisson regression with offsets in tensorflow. Pretty baller right.

# %%
import numpy as np
import pandas as pd

# %%
np.random.seed(333)
n = 1000000
df = pd.DataFrame(
    np.random.normal(np.tile(np.arange(10), n), np.tile(np.arange(1, 11) * 2, n), (n * 10)).reshape((n, 10)) / 100)
wts = np.array([-0.1, -0.2, 0, 0, 0.1, 0.2, 0.3, 0.0, 0.25, 0.5])
df['y'] = (df.values * wts).sum(axis=1)
df['rate'] = np.exp(df['y'])

# %%
# Did I really need to show this, with jupytext no!!!
df.describe()

# %%
# %matplotlib inline
df.rate.hist()

# %%
df['offset'] = np.random.randint(1, 10, n)

# %%
(np.random.poisson(df['rate'], (10, n)).T.mean(axis=1) - df.rate).hist()

# %%
df['cnt'] = np.random.poisson(df['rate'])
df.cnt.hist()

# %%
import statsmodels.api as sm


mod = sm.GLM(df['cnt'], df[np.arange(10)], family=sm.families.Poisson())

mod = mod.fit()

mod.summary()

# %%
df['cnt'] = [np.random.poisson(rate, offset).sum() for rate, offset in zip(df['rate'], df['offset'])]

# %%
df.cnt.hist()

# %%
import statsmodels.api as sm


mod = sm.GLM(df['cnt'], df[np.arange(10)], offset=np.log(df['offset']), family=sm.families.Poisson())

mod = mod.fit()

mod.summary()

# %%
import statsmodels.api as sm


mod = sm.GLM(df['cnt'] / df['offset'], df[np.arange(10)], family=sm.families.Poisson())

mod = mod.fit()

mod.summary()

# %%
from pyglmnet import GLM

# create an instance of the GLM class
glm = GLM(distr='poisson')
glm = glm.fit(df[np.arange(10)].values, df['cnt'].values/df['offset'].values)
glm

# %%
glm.get_params()

# %%
import keras

inl = keras.layers.Input((10,))
out = keras.layers.Dense(1, use_bias=False)(inl)
out = keras.layers.Lambda(lambda x: keras.backend.exp(x))(out)
model = keras.models.Model(inl, out)

model.compile(keras.optimizers.Adam(1e-3), 'poisson')
model.summary()
model.fit(df[np.arange(10)], df['cnt']/df['offset'], verbose=1)

# %%
model.get_weights()[0].ravel()

# %%


# %%
import keras

inl = keras.layers.Input((10,))
out = keras.layers.Dense(1, use_bias=False)(inl)
off = keras.layers.Input((1,))
out = keras.layers.add([out, off])
out = keras.layers.Lambda(lambda x: keras.backend.exp(x))(out)

model = keras.models.Model([inl, off], out)

model.compile(keras.optimizers.Adam(1e-3), 'poisson')
model.summary()
model.fit([df[np.arange(10)], np.log(df['offset'])], df['cnt'], verbose=1)

# %%
model.get_weights()[0].ravel()

# %%
# now with unequal number of exposures we need to use the offsets correctly to get the right answer

# %%
offsets = np.arange(1, 101)
import itertools
offsets=np.array(list(itertools.chain.from_iterable([np.repeat(i, off) for i, off in enumerate(offsets)])))
X=np.array([x/100 for x in offsets])[:, np.newaxis]
X.shape, offsets.shape

# %%
y=np.random.poisson(np.exp(X*1.2+.33)).ravel()
y.shape

# %%
mod = sm.GLM(y, sm.add_constant(X), family=sm.families.Poisson())

mod = mod.fit()

mod.summary()

# %%
Xg=pd.DataFrame(X).groupby(offsets).mean().values
yg=pd.DataFrame(y).groupby(offsets).sum().values.ravel()
Xg.shape, yg.shape

# %%
offsetsg = pd.DataFrame(offsets).groupby(offsets).size().values.ravel()

# %%
mod = sm.GLM(yg, sm.add_constant(Xg), offset=np.log(offsetsg), family=sm.families.Poisson())

mod = mod.fit()

mod.summary()

# %%
mod = sm.GLM(yg/offsetsg, sm.add_constant(Xg), family=sm.families.Poisson())

mod = mod.fit()
mod.summary()

# %%
# wow amazing dividing by exposures doesn't work!!!! Guess you actually have use the math to be correct :)

# %%
# bonus let's try tensorflow

# %%


# %%
import keras

inl = keras.layers.Input((2,))
out = keras.layers.Dense(1, use_bias=False)(inl)
off = keras.layers.Input((1,))
out = keras.layers.add([out, off])
out = keras.layers.Lambda(lambda x: keras.backend.exp(x))(out)

model = keras.models.Model([inl, off], out)

model.compile(keras.optimizers.SGD(1e-19), 'poisson')
model.summary()
model.fit([sm.add_constant(Xg), np.log(offsetsg)], yg, verbose=1, epochs=1)

# %%
model.get_weights()[0]

# %%

