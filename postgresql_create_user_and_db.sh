#!/usr/bin/bash

. ./.env

echo "************** CREATING DATABASE & USER *******************"

sudo su postgres <<EOF
createdb  $DATABASE;
psql -c "CREATE ROLE $USER;"
psql -c "ALTER ROLE $USER WITH LOGIN;"
psql -c "ALTER ROLE $USER WITH SUPERUSER;"
psql -c "ALTER USER $USER WITH PASSWORD '$PASSWORD';"
psql -c "ALTER ROLE $USER SET client_encoding TO 'utf8';"
psql -c "ALTER ROLE $USER SET default_transaction_isolation TO 'read committed';"
psql -c "ALTER ROLE $USER SET timezone TO 'UTC';"
psql -c "grant all privileges on database $DATABASE to $USER;"
psql -c "set role $USER;\c $DATABSE; create extension vector;"
exit
EOF

#echo "************** RESTARTING POSTGRESL *******************"
sudo service postgresql restart
