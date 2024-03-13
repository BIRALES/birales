db = connect("127.0.0.1:27017/birales");
db.createUser({ user: "birales_ro", pwd: password, roles: [{ role: "read", db: "birales" }] });
db.createUser({ user: "birales_admin", pwd: password, roles: [{ role: "dbAdmin", db: "birales" }] });
db.createUser({ user: "birales_owner", pwd: password, roles: [{ role: "dbOwner", db: "birales" }] });
db.createUser({ user: "birales_rw", pwd: password, roles: [{ role: "readWrite", db: "birales" }] });