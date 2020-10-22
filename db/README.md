## TODO 

To run the usual commands on manage.py, run `python3 manage.py` and enter command when prompted. Had to edit it out (see `manage.py`) due to errors with argv and configs. For running `runserver`, type the command twice (as prompted).

Commands to run when setting up from scratch.
```
makemigrations
migrate
createsuperuser
python manage.py syncdb

```
- I think tested, works fine. Just edit the front end ig.
+ Each env has it's own seperate table to store data.
+ 2 Apps :
  - `traj_db` : Contains the main meta-data DB to store the trajectories. Stores the language/instruction schema too.
  - `hindsight_instruction` : Web-App to play trajectories and store instructions.

+ Might migrate to Postgres in the future. Need sqllite-3 for speed now.
  - https://www.digitalocean.com/community/tutorials/how-to-use-postgresql-with-your-django-application-on-ubuntu-14-04
  - https://www.vphventures.com/how-to-migrate-your-django-project-from-sqlite-to-postgresql/

+ To run on Google Colabs
  - https://ngrok.com/
  - https://medium.com/@kshitijvijay271199/flask-on-google-colab-f6525986797b
  - https://stackoverflow.com/questions/59741453/is-there-a-general-way-to-run-web-applications-on-google-colab
  
