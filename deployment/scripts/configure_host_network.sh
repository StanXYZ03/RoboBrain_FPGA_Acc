#!/bin/sh
set -eu

HOST_IFACE="${HOST_IFACE:-enp4s0}"
HOST_ADDRESS="${HOST_ADDRESS:-192.168.10.1/24}"
[ "$(id -u)" = 0 ] || {
    echo "run with sudo: sudo $0" >&2
    exit 1
}
ip link set "$HOST_IFACE" up
ip address replace "$HOST_ADDRESS" dev "$HOST_IFACE"
ip -brief address show dev "$HOST_IFACE"
