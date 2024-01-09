## Port visits dataset

The dataset contains the visits of different vessels to worldwide ports. 
A visit is defined as the presence of a vessel in the area near a port. 
GPS data is taken from three years (2017-2019) of AIS messages. 

#### Port cluster
A port cluster is a set of ports in close proximity whose areas are merged and are treated as a single port. The largest port is chosen to be the cluster representative.
 
### Content of the dataset
* **visits_augmented.csv**: the list of visits as a csv file
* **port_mapping.csv**: port clusters mapping
* **WPI.shp**: shape file with the port data (open with geopandas)
* **WPI_Explanation_of_Data_Fields.pdf**: explanation of the field in the WPI port dataset
* **VesselTypeCodes2018.pdf**: AIS vessel types and categories


### Field description for the file visit_augmented.csv

* port: a numeric field with the WPI id of the port (or representative port of the cluster)
* mmsi: radio equipment identifier of the vessel
* uid: unique id that combines mssi and vessel type (radio equipment can be moved, vessels can change type)
* count: number of consecutive AIS messages for the vessel in the given port
* start: start date of the port visit
* end: end date of the port visit
* vesseltype: code type of the vessel
* imo: International Maritime Organization identification number of the vessel (https://en.wikipedia.org/wiki/IMO_number)
* callsign: call signs assigned as unique identifiers of the vessel (https://en.wikipedia.org/wiki/Maritime_call_sign) 
* duration_seconds: amount of seconds of visit duration (end - start)
* vessel_category: category of the vessel, according to the vesseltype. 
* country: country of the port
* latitude: latitude of the port
* longitude: longitude of the port
* port_name: name of the port
* region: continent or sub-continent of the port


