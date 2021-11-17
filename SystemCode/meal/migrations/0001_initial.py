# Generated by Django 3.2 on 2021-11-16 15:45

from django.db import migrations, models


class Migration(migrations.Migration):

    initial = True

    dependencies = [
    ]

    operations = [
        migrations.CreateModel(
            name='Meal',
            fields=[
                ('id', models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name='ID')),
                ('gender', models.CharField(max_length=1)),
                ('height', models.CharField(max_length=3)),
                ('weight', models.CharField(max_length=2)),
                ('age', models.CharField(max_length=3)),
                ('work_day', models.CharField(max_length=1)),
                ('muscle', models.CharField(max_length=1)),
            ],
        ),
    ]