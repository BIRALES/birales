#!/usr/bin/env bash

# Helper function to install required package
function install_package(){
    PKG_OK=$(dpkg-query -W --showformat='${Status}\n' $1 | grep "install ok installed")
    if [[ "" == "$PKG_OK" ]]; then
      echo "Installing $1."
      sudo DEBIAN_FRONTEND=noninteractive apt-get -qq --yes install $1 > /dev/null
      return  0 # Return success status
    else
      echo "$1 already installed"
      return 1  # Return fail status (already installed)
    fi
}

# Setup up MONGO database
# Check if mongo PPA is available, if not add it
if ! grep -q "^deb .*mongodb-org*" /etc/apt/sources.list /etc/apt/sources.list.d/*; then
    curl -fsSL https://pgp.mongodb.com/server-7.0.asc |  sudo gpg -o /usr/share/keyrings/mongodb-server-7.0.gpg --dearmor
    echo "deb [ arch=amd64,arm64 signed-by=/usr/share/keyrings/mongodb-server-7.0.gpg ] https://repo.mongodb.org/apt/ubuntu jammy/mongodb-org/7.0 multiverse" | sudo tee /etc/apt/sources.list.d/mongodb-org-7.0.list
    sudo apt -qq update
fi

# Install mongodb package
if install_package mongodb-org; then
    # Set up as a service
    if [[ ! -f /etc/systemd/system/mongodb.service ]]; then
        sudo cp mongodb.service /etc/systemd/system/
    fi

    # Create directory where mongo will store database
    if [[ ! -f /etc/systemd/system/mongodb.service ]]; then
      sudo mkdir -p /data/db/
      sudo chown `id -u` /data/db
    fi

    # Start up service
    sudo service mongod start

    # Check if mongo user password is defined in environment
    if [[ -z "$BIRALES__DB_PASSWORD" ]]; then
        RESULT=0
        while [[ "$RESULT" -eq 0 ]]; do
            echo -n "BIRALES password is not defined. Please enter a password: "
            read -s BIRALES__DB_PASSWORD
            echo
            echo -n "Re-enter password: "
            read -s PASSWORD2
            echo

            if [[ "$BIRALES__DB_PASSWORD" == "$PASSWORD2" ]]; then
                echo "Password set"
                RESULT=1
            else
                echo "Passwords do not match. Please re-try."
            fi
        done

        # Save password in .bashrc
        echo "export BIRALES__DB_PASSWORD=`echo $BIRALES__DB_PASSWORD`" >> ~/.bashrc
    fi

    # Create BIRALES root user
    mongosh 127.0.0.1:27017/admin --eval "db.createUser({user: 'birales_root', pwd: '$BIRALES__DB_PASSWORD', roles: ['root']})"

    # Create birales normal users
    mongosh --port 27017 -u "birales_root" -p "$BIRALES__DB_PASSWORD" --authenticationDatabase "admin" \
          --eval "var password='$BIRALES__DB_PASSWORD'" database_setup.js

    # Override mongo configuration file
    sudo cp mongod.conf /etc

    # Restart mongo service
    sudo service mongod restart

    # Done, source .bashrc
    source ~/.bashrc
fi