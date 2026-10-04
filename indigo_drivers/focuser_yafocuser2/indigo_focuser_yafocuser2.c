// Copyright (c) 2018 CloudMakers, s. r. o.
// All rights reserved.
//
// You can use this software under the terms of 'INDIGO Astronomy
// open-source license' (see LICENSE.md).
//
// THIS SOFTWARE IS PROVIDED BY THE AUTHORS 'AS IS' AND ANY EXPRESS
// OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
// WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY
// DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
// DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE
// GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
// INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
// NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
// SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

// version history
// 2.0 by Peter Polakovic <peter.polakovic@cloudmakers.eu>

/** INDIGO Yafocuser dual focuser driver
 \file indigo_focuser_yafocuser2.c
 */

#define DRIVER_VERSION 0x000A
#define DRIVER_NAME "indigo_focuser_yafocuser2"

#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <math.h>
#include <assert.h>
#include <errno.h>
#include <pthread.h>
#include <stdarg.h>

#include <sys/time.h>

#include <indigo/indigo_driver_xml.h>
#include <indigo/indigo_io.h>

#include "indigo_focuser_yafocuser2.h"

#define PRIVATE_DATA													((yafocuser2_private_data *)device->private_data)

typedef struct {
	int handle;
	indigo_timer *timer1;
	indigo_timer *timer2;
	pthread_mutex_t mutex;
} yafocuser2_private_data;

static bool yafocuser2_command(indigo_device *device, char *command, char *response, int max) {
	char c;
	struct timeval tv;
	indigo_write(PRIVATE_DATA->handle, command, strlen(command));
	if (response != NULL) {
		int index = 0;
		while (index < max) {
			fd_set readout;
			tv.tv_sec = 1;
			tv.tv_usec = 0;
			FD_ZERO(&readout);
			FD_SET(PRIVATE_DATA->handle, &readout);
			long result = select(PRIVATE_DATA->handle+1, &readout, NULL, NULL, &tv);
			if (result <= 0)
				break;
			result = read(PRIVATE_DATA->handle, &c, 1);
			if (result < 1) {
				INDIGO_DRIVER_ERROR(DRIVER_NAME, "Failed to read from %s -> %s (%d)", DEVICE_PORT_ITEM->text.value, strerror(errno), errno);
				return false;
			}
			if (c < 0)
				c = ':';
			if (c == '#')
				break;
			response[index++] = c;
		}
		response[index] = 0;
	}
	INDIGO_DRIVER_DEBUG(DRIVER_NAME, "Command '%s' -> '%s'", command, response != NULL ? response : "NULL");
	return true;
}

// -------------------------------------------------------------------------------- INDIGO focuser device implementation

static indigo_result focuser1_attach(indigo_device *device) {
	assert(device != NULL);
	assert(PRIVATE_DATA != NULL);
	if (indigo_focuser_attach(device, DRIVER_NAME, DRIVER_VERSION) == INDIGO_OK) {
		// -------------------------------------------------------------------------------- DEVICE_PORT, DEVICE_PORTS
		DEVICE_PORT_PROPERTY->hidden = false;
		DEVICE_PORTS_PROPERTY->hidden = false;
#ifdef INDIGO_MACOS
		for (int i = 0; i < DEVICE_PORTS_PROPERTY->count; i++) {
			if (!strncmp(DEVICE_PORTS_PROPERTY->items[i].name, "/dev/cu.usbmodem", 16)) {
				INDIGO_COPY_VALUE(DEVICE_PORT_ITEM->text.value, DEVICE_PORTS_PROPERTY->items[i].name);
				break;
			}
		}
#endif
#ifdef INDIGO_LINUX
		strcpy(DEVICE_PORT_ITEM->text.value, "/dev/ttyFocuser");
#endif
		// -------------------------------------------------------------------------------- INFO
		INFO_PROPERTY->count = 6;
		strcpy(INFO_DEVICE_MODEL_ITEM->text.value, "Yafocuser Dual Focuser");
		// -------------------------------------------------------------------------------- FOCUSER_TEMPERATURE
		FOCUSER_TEMPERATURE_PROPERTY->hidden = false;
		// -------------------------------------------------------------------------------- FOCUSER_SPEED
		FOCUSER_SPEED_ITEM->number.min = FOCUSER_SPEED_ITEM->number.value = FOCUSER_SPEED_ITEM->number.target = 1;
		FOCUSER_SPEED_ITEM->number.max = 5;
		FOCUSER_SPEED_ITEM->number.step = 1;
		// -------------------------------------------------------------------------------- FOCUSER_STEPS
		FOCUSER_STEPS_ITEM->number.min = 0;
		FOCUSER_STEPS_ITEM->number.max = 0xFFFF;
		FOCUSER_STEPS_ITEM->number.step = 1;
		// -------------------------------------------------------------------------------- FOCUSER_POSITION
		FOCUSER_POSITION_ITEM->number.min = 0;
		FOCUSER_POSITION_ITEM->number.max = 0xFFFF;
		FOCUSER_POSITION_ITEM->number.step = 1;
		// -------------------------------------------------------------------------------- FOCUSER_COMPENSATION
		FOCUSER_COMPENSATION_ITEM->number.min = -128;
		FOCUSER_COMPENSATION_ITEM->number.min = -127;
		FOCUSER_COMPENSATION_PROPERTY->hidden = false;
		FOCUSER_MODE_PROPERTY->hidden = false;
		// -------------------------------------------------------------------------------- FOCUSER_REVERSE_MOTION
		FOCUSER_REVERSE_MOTION_PROPERTY->hidden = false;
		// -------------------------------------------------------------------------------- FOCUSER_LIMITS
		FOCUSER_LIMITS_PROPERTY->hidden = false;
		FOCUSER_LIMITS_MIN_POSITION_ITEM->number.min = 0;
		FOCUSER_LIMITS_MIN_POSITION_ITEM->number.max = 0xFFFF;
		FOCUSER_LIMITS_MIN_POSITION_ITEM->number.step = 1;
		FOCUSER_LIMITS_MIN_POSITION_ITEM->number.value = FOCUSER_LIMITS_MIN_POSITION_ITEM->number.target = 0;
		FOCUSER_LIMITS_MAX_POSITION_ITEM->number.min = 0;
		FOCUSER_LIMITS_MAX_POSITION_ITEM->number.max = 0xFFFF;
		FOCUSER_LIMITS_MAX_POSITION_ITEM->number.step = 1;
		FOCUSER_LIMITS_MAX_POSITION_ITEM->number.value = FOCUSER_LIMITS_MAX_POSITION_ITEM->number.target = 0xFFFF;
		// --------------------------------------------------------------------------------
		ADDITIONAL_INSTANCES_PROPERTY->hidden = device->base_device != NULL;
		pthread_mutex_init(&PRIVATE_DATA->mutex, NULL);
		INDIGO_DEVICE_ATTACH_LOG(DRIVER_NAME, device->name);
		return indigo_focuser_enumerate_properties(device, NULL, NULL);
	}
	return INDIGO_FAILED;
}

static indigo_result focuser1_enumerate_properties(indigo_device *device, indigo_client *client, indigo_property *property) {
	return indigo_focuser_enumerate_properties(device, NULL, NULL);
}

static void focuser1_timer_callback(indigo_device *device) {
	if (!IS_CONNECTED)
		return;
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	static bool read_temperature = false;
	char response[16];
	if (read_temperature) {
		if (yafocuser2_command(device, ":GT#", response, sizeof(response))) {
			double temp = ((int8_t)strtol(response, NULL, 16)) / 2.0;
			if (FOCUSER_TEMPERATURE_ITEM->number.value != temp) {
				FOCUSER_TEMPERATURE_ITEM->number.value = temp;
				FOCUSER_TEMPERATURE_PROPERTY->state = INDIGO_OK_STATE;
				indigo_update_property(device, FOCUSER_TEMPERATURE_PROPERTY, NULL);
			}
		}
		read_temperature = false;
	} else {
		yafocuser2_command(device, ":C#", NULL, 0);
		read_temperature = true;
	}
	bool update = false;
	if (yafocuser2_command(device, ":F1GP#", response, sizeof(response))) {
		long pos = strtol(response, NULL, 16);
		if (FOCUSER_POSITION_ITEM->number.value != pos) {
			FOCUSER_POSITION_ITEM->number.value = pos;
			update = true;
		}
	}
	if (yafocuser2_command(device, ":F1GI#", response, sizeof(response))) {
		if (strcmp(response, "00") == 0) {
			if (FOCUSER_POSITION_PROPERTY->state == INDIGO_BUSY_STATE) {
				FOCUSER_STEPS_PROPERTY->state = INDIGO_OK_STATE;
				FOCUSER_POSITION_PROPERTY->state = INDIGO_OK_STATE;
				update = true;
			}
		} else {
			if (FOCUSER_POSITION_PROPERTY->state != INDIGO_BUSY_STATE) {
				FOCUSER_POSITION_PROPERTY->state = INDIGO_BUSY_STATE;
				FOCUSER_STEPS_PROPERTY->state = INDIGO_BUSY_STATE;
				update = true;
			}
		}
	}
	if (update) {
		indigo_update_property(device, FOCUSER_POSITION_PROPERTY, NULL);
		indigo_update_property(device, FOCUSER_STEPS_PROPERTY, NULL);
	}
	indigo_reschedule_timer(device, FOCUSER_POSITION_PROPERTY->state == INDIGO_BUSY_STATE ? 0.2 : 1.0, &PRIVATE_DATA->timer1);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static void focuser1_connection_handler(indigo_device *device) {
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	char response[64];
	if (CONNECTION_CONNECTED_ITEM->sw.value) {
		PRIVATE_DATA->handle = indigo_open_serial_with_speed(DEVICE_PORT_ITEM->text.value, 9600);
		if (PRIVATE_DATA->handle > 0) {
			for (int i = 0; true; i++) {
				if (yafocuser2_command(device, ":GV#", response, sizeof(response)) && strlen(response) == 2) {
					INDIGO_DRIVER_LOG(DRIVER_NAME, "YaFocuser dual focuser %c.%c", response[0], response[1]);
					break;
				} else if (i < 5) {
					INDIGO_DRIVER_ERROR(DRIVER_NAME, "No reply from YaFocuser dual focuser - retrying");
					indigo_usleep(2 * ONE_SECOND_DELAY);
				} else {
					INDIGO_DRIVER_ERROR(DRIVER_NAME, "Yafocuser dual focuser not detected");
					close(PRIVATE_DATA->handle);
					PRIVATE_DATA->handle = 0;
					break;
				}
			}
		}
		if (PRIVATE_DATA->handle > 0) {
			yafocuser2_command(device, ":C#", NULL, 0);
			yafocuser2_command(device, ":F1FQ#", NULL, 0);
			yafocuser2_command(device, ":F1SF#", NULL, 0);
			yafocuser2_command(device, ":-#", NULL, 0);
			yafocuser2_command(device, ":F1SD02#", NULL, 0);
			indigo_usleep(750000);
			if (yafocuser2_command(device, ":GT#", response, sizeof(response))) {
				FOCUSER_TEMPERATURE_ITEM->number.value = ((int8_t)strtol(response, NULL, 16)) / 2.0;
			}
			if (yafocuser2_command(device, ":F1GP#", response, sizeof(response))) {
				FOCUSER_POSITION_ITEM->number.value = FOCUSER_POSITION_ITEM->number.target = strtol(response, NULL, 16);
			}
			if (yafocuser2_command(device, ":GC#", response, sizeof(response))) {
				FOCUSER_COMPENSATION_ITEM->number.value = (char)strtol(response, NULL, 16);
			}
		}
		if (PRIVATE_DATA->handle > 0) {
			INDIGO_DRIVER_LOG(DRIVER_NAME, "Connected to %s", DEVICE_PORT_ITEM->text.value);
			indigo_set_timer(device, 0, focuser1_timer_callback, &PRIVATE_DATA->timer1);
			CONNECTION_PROPERTY->state = INDIGO_OK_STATE;
		} else {
			INDIGO_DRIVER_ERROR(DRIVER_NAME, "Failed to connect to %s", DEVICE_PORT_ITEM->text.value);
			CONNECTION_PROPERTY->state = INDIGO_ALERT_STATE;
			indigo_set_switch(CONNECTION_PROPERTY, CONNECTION_DISCONNECTED_ITEM, true);
		}
	} else {
		if (PRIVATE_DATA->handle > 0) {
			indigo_cancel_timer_sync(device, &PRIVATE_DATA->timer1);
			yafocuser2_command(device, ":F1FQ#", NULL, 0);
			INDIGO_DRIVER_LOG(DRIVER_NAME, "Disconnected");
			close(PRIVATE_DATA->handle);
			PRIVATE_DATA->handle = 0;
		}
		CONNECTION_PROPERTY->state = INDIGO_OK_STATE;
	}
	indigo_focuser_change_property(device, NULL, CONNECTION_PROPERTY);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static void focuser1_speed_handler(indigo_device *device) {
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	char command[16];
	snprintf(command, sizeof(command), ":F1SD%02X#", 0x01 << (int)FOCUSER_SPEED_ITEM->number.value);
	if (yafocuser2_command(device, command, NULL, 0)) {
		FOCUSER_SPEED_PROPERTY->state = INDIGO_OK_STATE;
	} else {
		FOCUSER_SPEED_PROPERTY->state = INDIGO_ALERT_STATE;
	}
	indigo_update_property(device, FOCUSER_SPEED_PROPERTY, NULL);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static void focuser1_steps_handler(indigo_device *device) {
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	char command[16];
	int position = FOCUSER_POSITION_ITEM->number.value + (int)FOCUSER_STEPS_ITEM->number.value * (FOCUSER_DIRECTION_MOVE_INWARD_ITEM->sw.value ? 1 : -1) * (FOCUSER_REVERSE_MOTION_ENABLED_ITEM->sw.value ? -1 : 1);
	if (position < 0)
		position = 0;
	if (position < FOCUSER_LIMITS_MIN_POSITION_ITEM->number.value)
		position = FOCUSER_LIMITS_MIN_POSITION_ITEM->number.value;
	if (position > 0xFFFF)
		position = 0xFFFF;
	if (position > FOCUSER_LIMITS_MAX_POSITION_ITEM->number.value)
		position = FOCUSER_LIMITS_MAX_POSITION_ITEM->number.value;
	snprintf(command, sizeof(command), ":F1SN%04X#:FG#", position);
	if (yafocuser2_command(device, command, NULL, 0)) {
		FOCUSER_POSITION_PROPERTY->state = INDIGO_BUSY_STATE;
	} else {
		FOCUSER_STEPS_PROPERTY->state = INDIGO_ALERT_STATE;
		FOCUSER_POSITION_PROPERTY->state = INDIGO_ALERT_STATE;
	}
	indigo_update_property(device, FOCUSER_STEPS_PROPERTY, NULL);
	indigo_update_property(device, FOCUSER_POSITION_PROPERTY, NULL);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static void focuser1_position_handler(indigo_device *device) {
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	char command[16];
	int position = (int)FOCUSER_POSITION_ITEM->number.target;
	if (position < FOCUSER_LIMITS_MIN_POSITION_ITEM->number.value)
		position = FOCUSER_LIMITS_MIN_POSITION_ITEM->number.value;
	if (position > FOCUSER_LIMITS_MAX_POSITION_ITEM->number.value)
		position = FOCUSER_LIMITS_MAX_POSITION_ITEM->number.value;
	FOCUSER_POSITION_ITEM->number.target = position;
	snprintf(command, sizeof(command), ":F1SN%04X#:FG#", position);
	if (yafocuser2_command(device, command, NULL, 0)) {
		FOCUSER_STEPS_PROPERTY->state = INDIGO_BUSY_STATE;
	} else {
		FOCUSER_POSITION_PROPERTY->state = INDIGO_ALERT_STATE;
		FOCUSER_STEPS_PROPERTY->state = INDIGO_ALERT_STATE;
	}
	indigo_update_property(device, FOCUSER_POSITION_PROPERTY, NULL);
	indigo_update_property(device, FOCUSER_STEPS_PROPERTY, NULL);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static void focuser1_abort_handler(indigo_device *device) {
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	if (FOCUSER_ABORT_MOTION_ITEM->sw.value) {
		FOCUSER_ABORT_MOTION_ITEM->sw.value = false;
		if (yafocuser2_command(device, ":F1FQ#", NULL, 0)) {
			FOCUSER_ABORT_MOTION_PROPERTY->state = INDIGO_OK_STATE;
			FOCUSER_POSITION_PROPERTY->state = INDIGO_ALERT_STATE;
			FOCUSER_STEPS_PROPERTY->state = INDIGO_ALERT_STATE;
			indigo_update_property(device, FOCUSER_POSITION_PROPERTY, NULL);
			indigo_update_property(device, FOCUSER_STEPS_PROPERTY, NULL);
		} else {
			FOCUSER_ABORT_MOTION_PROPERTY->state = INDIGO_ALERT_STATE;
		}
	} else {
		FOCUSER_ABORT_MOTION_PROPERTY->state = INDIGO_OK_STATE;
	}
	indigo_update_property(device, FOCUSER_ABORT_MOTION_PROPERTY, NULL);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static void focuser1_mode_handler(indigo_device *device) {
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	if (yafocuser2_command(device, FOCUSER_MODE_AUTOMATIC_ITEM->sw.value ? ":+#" : ":-#", NULL, 0)) {
		FOCUSER_MODE_PROPERTY->state = INDIGO_OK_STATE;
	} else {
		FOCUSER_MODE_PROPERTY->state = INDIGO_ALERT_STATE;
	}
	indigo_update_property(device, FOCUSER_MODE_PROPERTY, NULL);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static void focuser1_compensation_handler(indigo_device *device) {
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	char command[16];
	sprintf(command, ":SC%02X#", ((char)FOCUSER_COMPENSATION_ITEM->number.value) & 0xFF);
	if (yafocuser2_command(device, command, NULL, 0)) {
		FOCUSER_COMPENSATION_PROPERTY->state = INDIGO_OK_STATE;
	} else {
		FOCUSER_COMPENSATION_PROPERTY->state = INDIGO_ALERT_STATE;
	}
	indigo_update_property(device, FOCUSER_COMPENSATION_PROPERTY, NULL);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static indigo_result focuser1_change_property(indigo_device *device, indigo_client *client, indigo_property *property) {
	assert(device != NULL);
	assert(DEVICE_CONTEXT != NULL);
	assert(property != NULL);
	if (indigo_property_match_changeable(CONNECTION_PROPERTY, property)) {
		// -------------------------------------------------------------------------------- CONNECTION
		if (indigo_ignore_connection_change(device, property))
			return INDIGO_OK;
		indigo_property_copy_values(CONNECTION_PROPERTY, property, false);
		CONNECTION_PROPERTY->state = INDIGO_BUSY_STATE;
		indigo_update_property(device, CONNECTION_PROPERTY, NULL);
		indigo_set_timer(device, 0, focuser1_connection_handler, NULL);
		return INDIGO_OK;
	} else if (indigo_property_match_changeable(FOCUSER_SPEED_PROPERTY, property)) {
		// -------------------------------------------------------------------------------- FOCUSER_SPEED
		indigo_property_copy_values(FOCUSER_SPEED_PROPERTY, property, false);
		FOCUSER_SPEED_PROPERTY->state = INDIGO_BUSY_STATE;
		indigo_update_property(device, FOCUSER_SPEED_PROPERTY, NULL);
		indigo_set_timer(device, 0, focuser1_speed_handler, NULL);
		return INDIGO_OK;
	} else if (indigo_property_match_changeable(FOCUSER_STEPS_PROPERTY, property)) {
		// -------------------------------------------------------------------------------- FOCUSER_STEPS
		indigo_property_copy_values(FOCUSER_STEPS_PROPERTY, property, false);
		FOCUSER_STEPS_PROPERTY->state = INDIGO_BUSY_STATE;
		indigo_update_property(device, FOCUSER_STEPS_PROPERTY, NULL);
		indigo_set_timer(device, 0, focuser1_steps_handler, NULL);
		return INDIGO_OK;
	} else if (indigo_property_match_changeable(FOCUSER_POSITION_PROPERTY, property)) {
		// -------------------------------------------------------------------------------- FOCUSER_POSITION
		long position = FOCUSER_POSITION_ITEM->number.value;
		indigo_property_copy_values(FOCUSER_POSITION_PROPERTY, property, false);
		FOCUSER_POSITION_ITEM->number.value = position;
		FOCUSER_POSITION_PROPERTY->state = INDIGO_BUSY_STATE;
		indigo_update_property(device, FOCUSER_POSITION_PROPERTY, NULL);
		indigo_set_timer(device, 0, focuser1_position_handler, NULL);
		return INDIGO_OK;
	} else if (indigo_property_match_changeable(FOCUSER_ABORT_MOTION_PROPERTY, property)) {
		// -------------------------------------------------------------------------------- FOCUSER_ABORT_MOTION
		indigo_property_copy_values(FOCUSER_ABORT_MOTION_PROPERTY, property, false);
		FOCUSER_ABORT_MOTION_PROPERTY->state = INDIGO_BUSY_STATE;
		indigo_update_property(device, FOCUSER_ABORT_MOTION_PROPERTY, NULL);
		indigo_set_timer(device, 0, focuser1_abort_handler, NULL);
		return INDIGO_OK;
		// -------------------------------------------------------------------------------- FOCUSER_MODE
	} else if (indigo_property_match_changeable(FOCUSER_MODE_PROPERTY, property)) {
		indigo_property_copy_values(FOCUSER_MODE_PROPERTY, property, false);
		FOCUSER_MODE_PROPERTY->state = INDIGO_BUSY_STATE;
		indigo_update_property(device, FOCUSER_POSITION_PROPERTY, NULL);
		indigo_set_timer(device, 0, focuser1_mode_handler, NULL);
		return INDIGO_OK;
		// -------------------------------------------------------------------------------- FOCUSER_COMPENSATION
	} else if (indigo_property_match_changeable(FOCUSER_COMPENSATION_PROPERTY, property)) {
		indigo_property_copy_values(FOCUSER_COMPENSATION_PROPERTY, property, false);
		FOCUSER_COMPENSATION_PROPERTY->state = INDIGO_BUSY_STATE;
		indigo_update_property(device, FOCUSER_COMPENSATION_PROPERTY, NULL);
		indigo_set_timer(device, 0, focuser1_compensation_handler, NULL);
		return INDIGO_OK;
	}
	return indigo_focuser_change_property(device, client, property);
}

static indigo_result focuser1_detach(indigo_device *device) {
	assert(device != NULL);
	if (IS_CONNECTED) {
		indigo_set_switch(CONNECTION_PROPERTY, CONNECTION_DISCONNECTED_ITEM, true);
		focuser1_connection_handler(device);
	}
	pthread_mutex_destroy(&PRIVATE_DATA->mutex);
	INDIGO_DEVICE_DETACH_LOG(DRIVER_NAME, device->name);
	return indigo_focuser_detach(device);
}

// ----------------------------------------------------------------------------------- Focuser 2

static indigo_result focuser2_attach(indigo_device *device) {
	assert(device != NULL);
	assert(PRIVATE_DATA != NULL);
	if (indigo_focuser_attach(device, DRIVER_NAME, DRIVER_VERSION) == INDIGO_OK) {
		// -------------------------------------------------------------------------------- DEVICE_PORT, DEVICE_PORTS
		DEVICE_PORT_PROPERTY->hidden = false;
		DEVICE_PORTS_PROPERTY->hidden = false;
#ifdef INDIGO_MACOS
		for (int i = 0; i < DEVICE_PORTS_PROPERTY->count; i++) {
			if (!strncmp(DEVICE_PORTS_PROPERTY->items[i].name, "/dev/cu.usbmodem", 16)) {
				INDIGO_COPY_VALUE(DEVICE_PORT_ITEM->text.value, DEVICE_PORTS_PROPERTY->items[i].name);
				break;
			}
		}
#endif
#ifdef INDIGO_LINUX
		strcpy(DEVICE_PORT_ITEM->text.value, "/dev/ttyFocuser");
#endif
		// -------------------------------------------------------------------------------- INFO
		INFO_PROPERTY->count = 6;
		strcpy(INFO_DEVICE_MODEL_ITEM->text.value, "Yafocuser Dual Focuser");
		// -------------------------------------------------------------------------------- FOCUSER_TEMPERATURE
		FOCUSER_TEMPERATURE_PROPERTY->hidden = false;
		// -------------------------------------------------------------------------------- FOCUSER_SPEED
		FOCUSER_SPEED_ITEM->number.min = FOCUSER_SPEED_ITEM->number.value = FOCUSER_SPEED_ITEM->number.target = 1;
		FOCUSER_SPEED_ITEM->number.max = 5;
		FOCUSER_SPEED_ITEM->number.step = 1;
		// -------------------------------------------------------------------------------- FOCUSER_STEPS
		FOCUSER_STEPS_ITEM->number.min = 0;
		FOCUSER_STEPS_ITEM->number.max = 0xFFFF;
		FOCUSER_STEPS_ITEM->number.step = 1;
		// -------------------------------------------------------------------------------- FOCUSER_POSITION
		FOCUSER_POSITION_ITEM->number.min = 0;
		FOCUSER_POSITION_ITEM->number.max = 0xFFFF;
		FOCUSER_POSITION_ITEM->number.step = 1;
		// -------------------------------------------------------------------------------- FOCUSER_COMPENSATION
		FOCUSER_COMPENSATION_ITEM->number.min = -128;
		FOCUSER_COMPENSATION_ITEM->number.min = -127;
		FOCUSER_COMPENSATION_PROPERTY->hidden = false;
		FOCUSER_MODE_PROPERTY->hidden = false;
		// -------------------------------------------------------------------------------- FOCUSER_REVERSE_MOTION
		FOCUSER_REVERSE_MOTION_PROPERTY->hidden = false;
		// -------------------------------------------------------------------------------- FOCUSER_LIMITS
		FOCUSER_LIMITS_PROPERTY->hidden = false;
		FOCUSER_LIMITS_MIN_POSITION_ITEM->number.min = 0;
		FOCUSER_LIMITS_MIN_POSITION_ITEM->number.max = 0xFFFF;
		FOCUSER_LIMITS_MIN_POSITION_ITEM->number.step = 1;
		FOCUSER_LIMITS_MIN_POSITION_ITEM->number.value = FOCUSER_LIMITS_MIN_POSITION_ITEM->number.target = 0;
		FOCUSER_LIMITS_MAX_POSITION_ITEM->number.min = 0;
		FOCUSER_LIMITS_MAX_POSITION_ITEM->number.max = 0xFFFF;
		FOCUSER_LIMITS_MAX_POSITION_ITEM->number.step = 1;
		FOCUSER_LIMITS_MAX_POSITION_ITEM->number.value = FOCUSER_LIMITS_MAX_POSITION_ITEM->number.target = 0xFFFF;
		// --------------------------------------------------------------------------------
		ADDITIONAL_INSTANCES_PROPERTY->hidden = device->base_device != NULL;
		pthread_mutex_init(&PRIVATE_DATA->mutex, NULL);
		INDIGO_DEVICE_ATTACH_LOG(DRIVER_NAME, device->name);
		return indigo_focuser_enumerate_properties(device, NULL, NULL);
	}
	return INDIGO_FAILED;
}

static indigo_result focuser2_enumerate_properties(indigo_device *device, indigo_client *client, indigo_property *property) {
	return indigo_focuser_enumerate_properties(device, NULL, NULL);
}

static void focuser2_timer_callback(indigo_device *device) {
	if (!IS_CONNECTED)
		return;
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	static bool read_temperature = false;
	char response[16];
	if (read_temperature) {
		if (yafocuser2_command(device, ":GT#", response, sizeof(response))) {
			double temp = ((int8_t)strtol(response, NULL, 16)) / 2.0;
			if (FOCUSER_TEMPERATURE_ITEM->number.value != temp) {
				FOCUSER_TEMPERATURE_ITEM->number.value = temp;
				FOCUSER_TEMPERATURE_PROPERTY->state = INDIGO_OK_STATE;
				indigo_update_property(device, FOCUSER_TEMPERATURE_PROPERTY, NULL);
			}
		}
		read_temperature = false;
	} else {
		yafocuser2_command(device, ":C#", NULL, 0);
		read_temperature = true;
	}
	bool update = false;
	if (yafocuser2_command(device, ":F2GP#", response, sizeof(response))) {
		long pos = strtol(response, NULL, 16);
		if (FOCUSER_POSITION_ITEM->number.value != pos) {
			FOCUSER_POSITION_ITEM->number.value = pos;
			update = true;
		}
	}
	if (yafocuser2_command(device, ":F2GI#", response, sizeof(response))) {
		if (strcmp(response, "00") == 0) {
			if (FOCUSER_POSITION_PROPERTY->state == INDIGO_BUSY_STATE) {
				FOCUSER_STEPS_PROPERTY->state = INDIGO_OK_STATE;
				FOCUSER_POSITION_PROPERTY->state = INDIGO_OK_STATE;
				update = true;
			}
		} else {
			if (FOCUSER_POSITION_PROPERTY->state != INDIGO_BUSY_STATE) {
				FOCUSER_POSITION_PROPERTY->state = INDIGO_BUSY_STATE;
				FOCUSER_STEPS_PROPERTY->state = INDIGO_BUSY_STATE;
				update = true;
			}
		}
	}
	if (update) {
		indigo_update_property(device, FOCUSER_POSITION_PROPERTY, NULL);
		indigo_update_property(device, FOCUSER_STEPS_PROPERTY, NULL);
	}
	indigo_reschedule_timer(device, FOCUSER_POSITION_PROPERTY->state == INDIGO_BUSY_STATE ? 0.2 : 1.0, &PRIVATE_DATA->timer2);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static void focuser2_connection_handler(indigo_device *device) {
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	char response[64];
	if (CONNECTION_CONNECTED_ITEM->sw.value) {
		PRIVATE_DATA->handle = indigo_open_serial_with_speed(DEVICE_PORT_ITEM->text.value, 9600);
		if (PRIVATE_DATA->handle > 0) {
			for (int i = 0; true; i++) {
				if (yafocuser2_command(device, ":GV#", response, sizeof(response)) && strlen(response) == 2) {
					INDIGO_DRIVER_LOG(DRIVER_NAME, "YaFocuser dual focuser %c.%c", response[0], response[1]);
					break;
				} else if (i < 5) {
					INDIGO_DRIVER_ERROR(DRIVER_NAME, "No reply from YaFocuser dual focuser - retrying");
					indigo_usleep(2 * ONE_SECOND_DELAY);
				} else {
					INDIGO_DRIVER_ERROR(DRIVER_NAME, "Yafocuser dual focuser not detected");
					close(PRIVATE_DATA->handle);
					PRIVATE_DATA->handle = 0;
					break;
				}
			}
		}
		if (PRIVATE_DATA->handle > 0) {
			yafocuser2_command(device, ":C#", NULL, 0);
			yafocuser2_command(device, ":F2FQ#", NULL, 0);
			yafocuser2_command(device, ":F2SF#", NULL, 0);
			yafocuser2_command(device, ":-#", NULL, 0);
			yafocuser2_command(device, ":F2SD02#", NULL, 0);
			indigo_usleep(750000);
			if (yafocuser2_command(device, ":GT#", response, sizeof(response))) {
				FOCUSER_TEMPERATURE_ITEM->number.value = ((int8_t)strtol(response, NULL, 16)) / 2.0;
			}
			if (yafocuser2_command(device, ":F2GP#", response, sizeof(response))) {
				FOCUSER_POSITION_ITEM->number.value = FOCUSER_POSITION_ITEM->number.target = strtol(response, NULL, 16);
			}
			if (yafocuser2_command(device, ":GC#", response, sizeof(response))) {
				FOCUSER_COMPENSATION_ITEM->number.value = (char)strtol(response, NULL, 16);
			}
		}
		if (PRIVATE_DATA->handle > 0) {
			INDIGO_DRIVER_LOG(DRIVER_NAME, "Connected to %s", DEVICE_PORT_ITEM->text.value);
			indigo_set_timer(device, 0, focuser2_timer_callback, &PRIVATE_DATA->timer2);
			CONNECTION_PROPERTY->state = INDIGO_OK_STATE;
		} else {
			INDIGO_DRIVER_ERROR(DRIVER_NAME, "Failed to connect to %s", DEVICE_PORT_ITEM->text.value);
			CONNECTION_PROPERTY->state = INDIGO_ALERT_STATE;
			indigo_set_switch(CONNECTION_PROPERTY, CONNECTION_DISCONNECTED_ITEM, true);
		}
	} else {
		if (PRIVATE_DATA->handle > 0) {
			indigo_cancel_timer_sync(device, &PRIVATE_DATA->timer2);
			yafocuser2_command(device, ":F2FQ#", NULL, 0);
			INDIGO_DRIVER_LOG(DRIVER_NAME, "Disconnected");
			close(PRIVATE_DATA->handle);
			PRIVATE_DATA->handle = 0;
		}
		CONNECTION_PROPERTY->state = INDIGO_OK_STATE;
	}
	indigo_focuser_change_property(device, NULL, CONNECTION_PROPERTY);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static void focuser2_speed_handler(indigo_device *device) {
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	char command[16];
	snprintf(command, sizeof(command), ":F2SD%02X#", 0x01 << (int)FOCUSER_SPEED_ITEM->number.value);
	if (yafocuser2_command(device, command, NULL, 0)) {
		FOCUSER_SPEED_PROPERTY->state = INDIGO_OK_STATE;
	} else {
		FOCUSER_SPEED_PROPERTY->state = INDIGO_ALERT_STATE;
	}
	indigo_update_property(device, FOCUSER_SPEED_PROPERTY, NULL);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static void focuser2_steps_handler(indigo_device *device) {
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	char command[16];
	int position = FOCUSER_POSITION_ITEM->number.value + (int)FOCUSER_STEPS_ITEM->number.value * (FOCUSER_DIRECTION_MOVE_INWARD_ITEM->sw.value ? 1 : -1) * (FOCUSER_REVERSE_MOTION_ENABLED_ITEM->sw.value ? -1 : 1);
	if (position < 0)
		position = 0;
	if (position < FOCUSER_LIMITS_MIN_POSITION_ITEM->number.value)
		position = FOCUSER_LIMITS_MIN_POSITION_ITEM->number.value;
	if (position > 0xFFFF)
		position = 0xFFFF;
	if (position > FOCUSER_LIMITS_MAX_POSITION_ITEM->number.value)
		position = FOCUSER_LIMITS_MAX_POSITION_ITEM->number.value;
	snprintf(command, sizeof(command), ":F2SN%04X#:FG#", position);
	if (yafocuser2_command(device, command, NULL, 0)) {
		FOCUSER_POSITION_PROPERTY->state = INDIGO_BUSY_STATE;
	} else {
		FOCUSER_STEPS_PROPERTY->state = INDIGO_ALERT_STATE;
		FOCUSER_POSITION_PROPERTY->state = INDIGO_ALERT_STATE;
	}
	indigo_update_property(device, FOCUSER_STEPS_PROPERTY, NULL);
	indigo_update_property(device, FOCUSER_POSITION_PROPERTY, NULL);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static void focuser2_position_handler(indigo_device *device) {
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	char command[16];
	int position = (int)FOCUSER_POSITION_ITEM->number.target;
	if (position < FOCUSER_LIMITS_MIN_POSITION_ITEM->number.value)
		position = FOCUSER_LIMITS_MIN_POSITION_ITEM->number.value;
	if (position > FOCUSER_LIMITS_MAX_POSITION_ITEM->number.value)
		position = FOCUSER_LIMITS_MAX_POSITION_ITEM->number.value;
	FOCUSER_POSITION_ITEM->number.target = position;
	snprintf(command, sizeof(command), ":F2SN%04X#:FG#", position);
	if (yafocuser2_command(device, command, NULL, 0)) {
		FOCUSER_STEPS_PROPERTY->state = INDIGO_BUSY_STATE;
	} else {
		FOCUSER_POSITION_PROPERTY->state = INDIGO_ALERT_STATE;
		FOCUSER_STEPS_PROPERTY->state = INDIGO_ALERT_STATE;
	}
	indigo_update_property(device, FOCUSER_POSITION_PROPERTY, NULL);
	indigo_update_property(device, FOCUSER_STEPS_PROPERTY, NULL);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static void focuser2_abort_handler(indigo_device *device) {
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	if (FOCUSER_ABORT_MOTION_ITEM->sw.value) {
		FOCUSER_ABORT_MOTION_ITEM->sw.value = false;
		if (yafocuser2_command(device, ":F2FQ#", NULL, 0)) {
			FOCUSER_ABORT_MOTION_PROPERTY->state = INDIGO_OK_STATE;
			FOCUSER_POSITION_PROPERTY->state = INDIGO_ALERT_STATE;
			FOCUSER_STEPS_PROPERTY->state = INDIGO_ALERT_STATE;
			indigo_update_property(device, FOCUSER_POSITION_PROPERTY, NULL);
			indigo_update_property(device, FOCUSER_STEPS_PROPERTY, NULL);
		} else {
			FOCUSER_ABORT_MOTION_PROPERTY->state = INDIGO_ALERT_STATE;
		}
	} else {
		FOCUSER_ABORT_MOTION_PROPERTY->state = INDIGO_OK_STATE;
	}
	indigo_update_property(device, FOCUSER_ABORT_MOTION_PROPERTY, NULL);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static void focuser2_mode_handler(indigo_device *device) {
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	if (yafocuser2_command(device, FOCUSER_MODE_AUTOMATIC_ITEM->sw.value ? ":+#" : ":-#", NULL, 0)) {
		FOCUSER_MODE_PROPERTY->state = INDIGO_OK_STATE;
	} else {
		FOCUSER_MODE_PROPERTY->state = INDIGO_ALERT_STATE;
	}
	indigo_update_property(device, FOCUSER_MODE_PROPERTY, NULL);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}

static void focuser2_compensation_handler(indigo_device *device) {
	pthread_mutex_lock(&PRIVATE_DATA->mutex);
	char command[16];
	sprintf(command, ":SC%02X#", ((char)FOCUSER_COMPENSATION_ITEM->number.value) & 0xFF);
	if (yafocuser2_command(device, command, NULL, 0)) {
		FOCUSER_COMPENSATION_PROPERTY->state = INDIGO_OK_STATE;
	} else {
		FOCUSER_COMPENSATION_PROPERTY->state = INDIGO_ALERT_STATE;
	}
	indigo_update_property(device, FOCUSER_COMPENSATION_PROPERTY, NULL);
	pthread_mutex_unlock(&PRIVATE_DATA->mutex);
}


static indigo_result focuser2_change_property(indigo_device *device, indigo_client *client, indigo_property *property) {
	assert(device != NULL);
	assert(DEVICE_CONTEXT != NULL);
	assert(property != NULL);
	if (indigo_property_match_changeable(CONNECTION_PROPERTY, property)) {
		// -------------------------------------------------------------------------------- CONNECTION
		if (indigo_ignore_connection_change(device, property))
			return INDIGO_OK;
		indigo_property_copy_values(CONNECTION_PROPERTY, property, false);
		CONNECTION_PROPERTY->state = INDIGO_BUSY_STATE;
		indigo_update_property(device, CONNECTION_PROPERTY, NULL);
		indigo_set_timer(device, 0, focuser2_connection_handler, NULL);
		return INDIGO_OK;
	} else if (indigo_property_match_changeable(FOCUSER_SPEED_PROPERTY, property)) {
		// -------------------------------------------------------------------------------- FOCUSER_SPEED
		indigo_property_copy_values(FOCUSER_SPEED_PROPERTY, property, false);
		FOCUSER_SPEED_PROPERTY->state = INDIGO_BUSY_STATE;
		indigo_update_property(device, FOCUSER_SPEED_PROPERTY, NULL);
		indigo_set_timer(device, 0, focuser2_speed_handler, NULL);
		return INDIGO_OK;
	} else if (indigo_property_match_changeable(FOCUSER_STEPS_PROPERTY, property)) {
		// -------------------------------------------------------------------------------- FOCUSER_STEPS
		indigo_property_copy_values(FOCUSER_STEPS_PROPERTY, property, false);
		FOCUSER_STEPS_PROPERTY->state = INDIGO_BUSY_STATE;
		indigo_update_property(device, FOCUSER_STEPS_PROPERTY, NULL);
		indigo_set_timer(device, 0, focuser2_steps_handler, NULL);
		return INDIGO_OK;
	} else if (indigo_property_match_changeable(FOCUSER_POSITION_PROPERTY, property)) {
		// -------------------------------------------------------------------------------- FOCUSER_POSITION
		long position = FOCUSER_POSITION_ITEM->number.value;
		indigo_property_copy_values(FOCUSER_POSITION_PROPERTY, property, false);
		FOCUSER_POSITION_ITEM->number.value = position;
		FOCUSER_POSITION_PROPERTY->state = INDIGO_BUSY_STATE;
		indigo_update_property(device, FOCUSER_POSITION_PROPERTY, NULL);
		indigo_set_timer(device, 0, focuser2_position_handler, NULL);
		return INDIGO_OK;
	} else if (indigo_property_match_changeable(FOCUSER_ABORT_MOTION_PROPERTY, property)) {
		// -------------------------------------------------------------------------------- FOCUSER_ABORT_MOTION
		indigo_property_copy_values(FOCUSER_ABORT_MOTION_PROPERTY, property, false);
		FOCUSER_ABORT_MOTION_PROPERTY->state = INDIGO_BUSY_STATE;
		indigo_update_property(device, FOCUSER_ABORT_MOTION_PROPERTY, NULL);
		indigo_set_timer(device, 0, focuser2_abort_handler, NULL);
		return INDIGO_OK;
		// -------------------------------------------------------------------------------- FOCUSER_MODE
	} else if (indigo_property_match_changeable(FOCUSER_MODE_PROPERTY, property)) {
		indigo_property_copy_values(FOCUSER_MODE_PROPERTY, property, false);
		FOCUSER_MODE_PROPERTY->state = INDIGO_BUSY_STATE;
		indigo_update_property(device, FOCUSER_POSITION_PROPERTY, NULL);
		indigo_set_timer(device, 0, focuser2_mode_handler, NULL);
		return INDIGO_OK;
		// -------------------------------------------------------------------------------- FOCUSER_COMPENSATION
	} else if (indigo_property_match_changeable(FOCUSER_COMPENSATION_PROPERTY, property)) {
		indigo_property_copy_values(FOCUSER_COMPENSATION_PROPERTY, property, false);
		FOCUSER_COMPENSATION_PROPERTY->state = INDIGO_BUSY_STATE;
		indigo_update_property(device, FOCUSER_COMPENSATION_PROPERTY, NULL);
		indigo_set_timer(device, 0, focuser2_compensation_handler, NULL);
		return INDIGO_OK;
	}
	return indigo_focuser_change_property(device, client, property);
}

static indigo_result focuser2_detach(indigo_device *device) {
	assert(device != NULL);
	if (IS_CONNECTED) {
		indigo_set_switch(CONNECTION_PROPERTY, CONNECTION_DISCONNECTED_ITEM, true);
		focuser2_connection_handler(device);
	}
	pthread_mutex_destroy(&PRIVATE_DATA->mutex);
	INDIGO_DEVICE_DETACH_LOG(DRIVER_NAME, device->name);
	return indigo_focuser_detach(device);
}


// -------------------------------------------------------------------------------- INDIGO driver implementation

indigo_result indigo_focuser_yafocuser2(indigo_driver_action action, indigo_driver_info *info) {
	static indigo_driver_action last_action = INDIGO_DRIVER_SHUTDOWN;
	static yafocuser2_private_data *private_data = NULL;
	static indigo_device *focuser1 = NULL;
	static indigo_device *focuser2 = NULL;

	static indigo_device focuser1_template = INDIGO_DEVICE_INITIALIZER(
		"Yafocuser 1",
		focuser1_attach,
		focuser1_enumerate_properties,
		focuser1_change_property,
		NULL,
		focuser1_detach
	);

	static indigo_device focuser2_template = INDIGO_DEVICE_INITIALIZER(
		"Yafocuser 2",
		focuser2_attach,
		focuser2_enumerate_properties,
		focuser2_change_property,
		NULL,
		focuser2_detach
	);

	SET_DRIVER_INFO(info, "Yafocuser dual Focuser", __FUNCTION__, DRIVER_VERSION, false, last_action);

	if (action == last_action)
		return INDIGO_OK;

	switch (action) {
		case INDIGO_DRIVER_INIT:
			last_action = action;
			private_data = indigo_safe_malloc(sizeof(yafocuser2_private_data));
			focuser1 = indigo_safe_malloc_copy(sizeof(indigo_device), &focuser1_template);
			focuser1->private_data = private_data;
			indigo_attach_device(focuser1);
			focuser2 = indigo_safe_malloc_copy(sizeof(indigo_device), &focuser2_template);
			focuser2->private_data = private_data;
			focuser2->master_device = focuser1;
			indigo_attach_device(focuser2);
			break;

		case INDIGO_DRIVER_SHUTDOWN:
			VERIFY_NOT_CONNECTED(focuser1);
			VERIFY_NOT_CONNECTED(focuser2);
			last_action = action;
			if (focuser1 != NULL) {
				indigo_detach_device(focuser1);
				free(focuser1);
				focuser1 = NULL;
			}
			if (focuser2 != NULL) {
				indigo_detach_device(focuser2);
				free(focuser2);
				focuser2 = NULL;
			}
			if (private_data != NULL) {
				free(private_data);
				private_data = NULL;
			}
			break;

		case INDIGO_DRIVER_INFO:
			break;
	}

	return INDIGO_OK;
}
