import matplotlib.pyplot as plt
import numpy as np

import Inference_close as I_close
import Inference_high as I_high
import Inference_low as I_low
import Inference_open as I_open

import Backtesting_close as B_close
import Backtesting_high as B_high
import Backtesting_low as B_low
import Backtesting_open as B_open

ic_y_real, ic_t_pred, ic_m_pred, ic_t_rmse, ic_m_rmse = I_close.Inference()
ih_y_real, ih_t_pred, ih_m_pred, ih_t_rmse, ih_m_rmse = I_high.Inference()
il_y_real, il_t_pred, il_m_pred, il_t_rmse, il_m_rmse = I_low.Inference()
io_y_real, io_t_pred, io_m_pred, io_t_rmse, io_m_rmse = I_open.Inference()

bc_y_real, bc_t_pred, bc_m_pred, bc_t_rmse, bc_m_rmse = B_close.Backtesting()
bh_y_real, bh_t_pred, bh_m_pred, bh_t_rmse, bh_m_rmse = B_high.Backtesting()
bl_y_real, bl_t_pred, bl_m_pred, bl_t_rmse, bl_m_rmse = B_low.Backtesting()
bo_y_real, bo_t_pred, bo_m_pred, bo_t_rmse, bo_m_rmse = B_open.Backtesting()


print('##################################################')
print('TSMC 2024/2/23 High Price Prediction: ' + str(bh_y_real[-1]))
print('Transformer Predicted High Price: ' + str(bh_t_pred[-1]))
print('Mamba Predicted High Price: ' + str(bh_m_pred[-1]))
print('Transformer RMSE: ' + str(bh_t_rmse))
print('Mamba RMSE: ' + str(bh_m_rmse))
print('##################################################')
print('TSMC 2024/2/23 Low Price Prediction: ' + str(bl_y_real[-1]))
print('Transformer Predicted Low Price: ' + str(bl_t_pred[-1]))
print('Mamba Predicted Low Price: ' + str(bl_m_pred[-1]))
print('Transformer RMSE: ' + str(bl_t_rmse))
print('Mamba RMSE: ' + str(bl_m_rmse))
print('##################################################')
print('TSMC 2024/2/23 Open Price Prediction: ' + str(bo_y_real[-1]))
print('Transformer Predicted Open Price: ' + str(bo_t_pred[-1]))
print('Mamba Predicted Open Price: ' + str(bo_m_pred[-1]))
print('Transformer RMSE: ' + str(bo_t_rmse))
print('Mamba RMSE: ' + str(bo_m_rmse))
print('##################################################')
print('TSMC 2024/2/23 Close Price Prediction: ' + str(bc_y_real[-1]))
print('Transformer Predicted Close Price: ' + str(bc_t_pred[-1]))
print('Mamba Predicted Close Price: ' + str(bc_m_pred[-1]))
print('Transformer RMSE: ' + str(bc_t_rmse))
print('Mamba RMSE: ' + str(bc_m_rmse))
print('##################################################')

plt.figure(figsize=(12, 8))

# High price prediction result
plt.subplot(2, 2, 1)
plt.plot(ih_y_real, label='real')
plt.plot(ih_t_pred, label='transformer')
plt.plot(ih_m_pred, label='mamba')
plt.xlabel('Transformer RMSE Error: {}'.format(ih_t_rmse) + "\n" + 'Mamba RMSE Error: {}'.format(ih_m_rmse))
plt.legend()
plt.title('High price prediction result')

# Low price prediction result
plt.subplot(2, 2, 2)
plt.plot(il_y_real, label='real')
plt.plot(il_t_pred, label='transformer')
plt.plot(il_m_pred, label='mamba')
plt.xlabel('Transformer RMSE Error: {}'.format(il_t_rmse) + "\n" + 'Mamba RMSE Error: {}'.format(il_m_rmse))
plt.legend()
plt.title('Low price prediction result')

# Open price prediction result
plt.subplot(2, 2, 3)
plt.plot(io_y_real, label='real')
plt.plot(io_t_pred, label='transformer')
plt.plot(io_m_pred, label='mamba')
plt.xlabel('Transformer RMSE Error: {}'.format(io_t_rmse) + "\n" + 'Mamba RMSE Error: {}'.format(io_m_rmse))
plt.legend()
plt.title('Open price prediction result')

# Close price prediction result
plt.subplot(2, 2, 4)
plt.plot(ic_y_real, label='real')
plt.plot(ic_t_pred, label='transformer')
plt.plot(ic_m_pred, label='mamba')
plt.xlabel('Transformer RMSE Error: {}'.format(ic_t_rmse) + "\n" + 'Mamba RMSE Error: {}'.format(ic_m_rmse))
plt.legend()
plt.title('Close price prediction result')

plt.tight_layout()
plt.show()