# import packages
from matplotlib.markers import MarkerStyle
from matplotlib.axes import Axes
import numpy as np
# import local implementations
from generators import Generator
from approximators import nn_approximator


def mse_test(approximator: nn_approximator,
             generator: Generator,
             lower=0.35, upper=2.00,
             num=100,
             seed=1):
    spots, pred_input, y_true, delta_true = generator.test_set(lower=lower, upper=upper, num=num, seed=seed)
    y_pred, xbar_pred = approximator.predict(spots)
    y_pred, xbar_pred = y_pred.reshape((-1, 1)), xbar_pred[:, 0]
    y_mse = ((y_pred - y_true) ** 2).mean(axis=0).item()
    xbar_mse = ((xbar_pred - delta_true[:, 0]) ** 2).mean(axis=0).item()
    return y_mse*1000, xbar_mse*1000


def plot_comparison(price_axs: Axes,
                    delta_axs: Axes,
                    model1: nn_approximator,
                    model2: nn_approximator,
                    generator: Generator,
                    lower=0.5,
                    upper=1.5,
                    num=100,
                    asset_idx=0,
                    plot_sims=True,
                    seed=None):
    if len(model1.y) != len(model2.y):
        raise Exception('Different number of samples in model 1 and model 2')
    n_samples = len(model1.y)
    spots, pred_input, y_true, delta_true = generator.test_set(lower=lower, upper=upper, num=num, seed=seed)

    for i, model in enumerate([model1, model2]):
        y_pred, xbar_pred = model.predict(spots)
        y_pred, xbar_pred = y_pred.reshape((-1, 1)), xbar_pred[:, asset_idx]
        y_mse = ((y_pred - y_true)**2).mean(axis=0).item()
        xbar_mse = ((xbar_pred - delta_true[:, asset_idx])**2).mean(axis=0).item()

        # Prepare textbox parameters
        price_textstr = '\n'.join((
            r'$MSE \ (x1000): %.3f$' % (y_mse*1000,),
            r'$Training \ time \ (s): %.3f$' % (model.training_time,)))
        delta_textstr = '\n'.join((
            r'$MSE \ (x1000): %.3f$' % (xbar_mse*1000,),
            r'$Training \ time \ (s): %.3f$' % (model.training_time,)))
        props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

        # Fill out price axes
        if plot_sims:
            price_axs0 = price_axs[i].scatter(model.x_raw[:, asset_idx], model.y_raw, c='lightgrey', s=8,
                                 linewidths=0.75,
                                 marker=MarkerStyle('x', fillstyle='none'),
                                 label='Simulations')
        price_axs1 = price_axs[i].scatter(pred_input, y_pred, c='lightsalmon', s=24, linewidths=0.75,
                                       marker=MarkerStyle('o', fillstyle='none'), label='Predicted Price')
        price_axs2 = price_axs[i].scatter(pred_input, y_true, c='red', label='True Price', s=1)
        price_axs[i].text(0.05, 0.95, price_textstr, transform=price_axs[i].transAxes, fontsize=8,
                verticalalignment='top', bbox=props)
        price_axs[i].tick_params(axis='both', labelsize=8)
        price_axs[i].set_xlim(lower, upper)
        # price_axs[i].set_ylim(top=y_true[-1]*1.1)
        price_axs[i].set_ylim(top=np.max(y_true)*1.1)

        # Fill out delta axes
        if model.differential and plot_sims: # only plot differential sims if model is of type differential
            delta_axs0 = delta_axs[i].scatter(model.x_raw[:, asset_idx], model.xbar_raw[:, asset_idx],
                                              c='lightgrey',
                                              s=8,
                                              linewidths=0.75,
                                              marker=MarkerStyle('x', fillstyle='none'),
                                              label='Simulations')
        delta_axs1 = delta_axs[i].scatter(pred_input, xbar_pred, c='lightblue', s=24, linewidths=0.75,
                                       marker=MarkerStyle('o', fillstyle='none'), label='Predicted Delta')
        delta_axs2 = delta_axs[i].scatter(pred_input, delta_true[:, asset_idx], c='blue', label='True Delta', s=1)
        delta_axs[i].text(0.05, 0.95, delta_textstr, transform=delta_axs[i].transAxes, fontsize=8,
                verticalalignment='top', bbox=props)
        delta_axs[i].tick_params(axis='both', labelsize=8)
        delta_axs[i].set_xlim(lower, upper)
        delta_axs[i].set_ylim(top=np.max(delta_true)*1.1)

    # Add number of samples and axes labels
    sample_textstr = '\n'.join((
        '{}'.format(n_samples),
        'training',
        'samples'))
    price_axs[1].text(1.1, 0.5, sample_textstr, transform=price_axs[1].transAxes, fontsize=10)
    delta_axs[1].text(1.1, 0.5, sample_textstr, transform=delta_axs[1].transAxes, fontsize=10)

    # Return handles for legend
    price_handles = [price_axs1, price_axs2]
    delta_handles = [delta_axs1, delta_axs2]
    if plot_sims:
        price_handles.append(price_axs0)
        delta_handles.append(delta_axs0)
    return price_handles, delta_handles


def stacked_comparison( axs: Axes,
                        model1,
                        model2,
                        generator: Generator,
                        labels=['Label1', 'Label2'],
                        type='Price',
                        lower=0.5,
                        upper=1.5,
                        num=100,
                        seed=None,
                        plot_sims=True,
                        scatter=False):
    spots, pred_input, y_true, delta_true = generator.test_set(lower=lower, upper=upper, num=num, seed=seed)
    y_pred_m1, xbar_pred_m1 = model1.predict(spots)
    y_pred_m2, xbar_pred_m2 = model2.predict(spots)
    if type == 'Price':
        pred_m1, pred_m2 = y_pred_m1, y_pred_m2
        sims = model1.y_raw
        true_label = y_true
    if type == 'Delta':
        pred_m1, pred_m2 = xbar_pred_m1[:, 0], xbar_pred_m2[:, 0]
        sims = model1.xbar_raw
        true_label = delta_true[:, 0]
    if plot_sims:
        sim_handle = axs.scatter(model1.x_raw, sims, c='lightgrey', s=8, linewidths=0.75,
                    marker=MarkerStyle('x', fillstyle='none'),
                    label='Simulations')
    if scatter:
        m1_handle = axs.scatter(pred_input, pred_m1, c='salmon', label=labels[0],
                                s=10, linewidths=0.75, marker=MarkerStyle('o', fillstyle='none'))
        m2_handle = axs.scatter(pred_input, pred_m2, c='dodgerblue', label=labels[1],
                                s=10, linewidths=0.75, marker=MarkerStyle('o', fillstyle='none'))
    else:
        m1_handle = axs.plot(pred_input, pred_m1, c='salmon', label=labels[0], lw=4)
        m2_handle = axs.plot(pred_input, pred_m2, c='dodgerblue', label=labels[1], lw=2)
        m1_handle = m1_handle[0]
        m2_handle = m2_handle[0]
    true_handle = axs.plot(pred_input, true_label, c='black', label='True', ls='--', lw=1)
    axs.tick_params(axis='both', labelsize=8)
    axs.set_xlim(lower, upper)
    axs.set_ylim(top=np.amax(true_label)*1.1, bottom=-0.1*np.amax(true_label))
    axs.set_xlabel('Spot')
    axs.set_ylabel(type)
    axs.set_title('Predicted {}'.format(type))
    handles = [m1_handle, m2_handle, true_handle[0]]
    if plot_sims:
        handles.append(sim_handle)
    return handles
