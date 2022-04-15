# brazilian-e-commerce
## Prerequisites
You have to configure the `settings.ini` file with the necessary credentials for the connection to the database.

Setup a python environment with virtualenv with `python=3.9` and activate it:

`$virtualenv env`

`$source env/Scripts/activate`

To install the necessary libraries for the project run:

`$pip install -r requirements.txt` 



## Extract raw data
To extract raw data from resource run:

`$python main.py`

To create tables on database run:

`$python script.py`
