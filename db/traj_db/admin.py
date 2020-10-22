from django.contrib import admin
from .models import ArchiveFile, SurrealRoboticsSuiteTrajectory, TrajectoryTag, RLBenchTrajectory
# Register your models here.

admin.site.register(RLBenchTrajectory)
admin.site.register(SurrealRoboticsSuiteTrajectory)
admin.site.register(ArchiveFile)
admin.site.register(TrajectoryTag)
