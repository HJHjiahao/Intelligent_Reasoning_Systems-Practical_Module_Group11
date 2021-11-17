# 引入表单类
from django import forms
# 引入文章模型
from .models import Meal

# 写文章的表单类
class MealForm(forms.ModelForm):
    class Meta:
        # 指明数据模型来源
        model = Meal
        # 定义表单包含的字段
        fields = ('gender','height','weight','age','work_day','muscle')