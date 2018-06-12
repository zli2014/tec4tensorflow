#-*-coding:UTF-8-*-

import numpy as np
import matplotlib.pyplot as plt
# x = np.linspace(0, 10, 1000)
# y = np.sin(x)
# plt.figure(figsize=(8,4))
# plt.plot(x,y,label="$sin(x)$",color="red",linewidth=2)
# plt.xlabel("Time(s)")
# plt.ylabel("Volt")
# plt.title("PyPlot First Example")
# plt.ylim(-1.2,1.2)
# plt.show()
# """
# 通过一系列函数设置当前Axes对象的各个属性：
# xlabel、ylabel：分别设置X、Y轴的标题文字。
# title：设置子图的标题。
# xlim、ylim：分别设置X、Y轴的显示范围。
# """
#
# """
# ===============================
# Legend using pre-defined labels
# ===============================
#
# Notice how the legend labels are defined with the plots!
# """
#


# Make some fake data.
a = b = np.arange(0, 3, .02)
c = np.exp(a)
d = c[::-1]
e = c+2
# Create plots with pre-defined labels.
fig, ax = plt.subplots()
ax.plot(a, c, 'k--', label='Model length')
ax.plot(a, d, 'k:', label='Data length')
ax.plot(a, e, 'k:', label='Data length')
# ax.plot(a, c + d, 'k', label='Total message length')

legend = ax.legend(loc='upper center', shadow=True, fontsize='x-large')

# Put a nicer background color on the legend.
legend.get_frame().set_facecolor('#00FFCC')

plt.show()