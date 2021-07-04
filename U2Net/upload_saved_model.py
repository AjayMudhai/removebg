import os, uuid
from azure.storage.blob import BlobServiceClient, BlobClient,ContentSettings, ContainerClient, __version__
import os
        
class AZStorage:
    def __init__(self):
        self.connection_string = 'DefaultEndpointsProtocol=https;AccountName=apimodel;AccountKey=Y+gW3VKqi6DdjFyNYxiv2hI6ZmMe5lngmVlTOte2MP70brdKcVN0b4qx8vk/3xtPoGlgP1ei0TOEhmewAba1Gg==;EndpointSuffix=core.windows.net'
        self.container_name='u2net'
        self.download_path='/home/azureuser/dataset/'
        self.upload_path='/home/azureuser/ai_overlap_cloth/ai_model/ACGPN_inference/sample'
    def make_container(self,name):
        ## Name may only contain lowercase letters, numbers, and hyphens, 
        ## and must begin with a letter or a number. 
        ## Each hyphen must be preceded and followed by a non-hyphen character.
        ## The name must also be between 3 and 63 characters long

        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        container_client = blob_service_client.get_container_client(name)
        container_client.create_container()


    def list_containers(self):
        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        try:
           
            all_containers = blob_service_client.list_containers(include_metadata=True)
            test_containers = blob_service_client.list_containers(name_starts_with='testing')
            for container in all_containers:
               print(container['name'], container['metadata'])
        except:
            print('Failed')


    def list_blobs(self,containerName):
        blob_service_client = BlobServiceClient.from_connection_string(self.connection_string)
        container_client = blob_service_client.get_container_client(containerName)
        try:
            for blob in container_client.list_blobs():
                print("Found blob: ", blob.name)
        except ResourceNotFoundError:
            print("Container not found.")


    def save_blob(self,file_name,file_content):
  
        download_file_path = os.path.join(self.download_path, file_name)
 
    
        os.makedirs(os.path.dirname(download_file_path), exist_ok=True)
 
        with open(download_file_path, "wb") as file:
            file.write(file_content)




        
    def download_blob(self,folderPath,fileType):
        self.blob_service_client =  BlobServiceClient.from_connection_string(self.connection_string)
        self.my_container = self.blob_service_client.get_container_client(self.container_name)
        print('Reading Files from Azure.Please Wait....')
        my_blobs = self.my_container.list_blobs(name_starts_with=folderPath)
        counter=0
    
        
        for blob in my_blobs:
          
            nl=blob.name.split('/')
            print(nl)
   
          
     
          
            n=nl[-1].split('.')
          
            if n[-1]== fileType:
    
            
                print('Downloading : {}'.format(blob.name))
                bytes = self.my_container.get_blob_client(blob).download_blob().readall()
                self.save_blob(blob.name, bytes)
                print('Done')

            else:
                print('Skipped')

    def upload_file(self,filename,file_path):
        try:
            self.blob_service_client =  BlobServiceClient.from_connection_string(self.connection_string)    
            self.my_container = self.blob_service_client.get_container_client(self.container_name)
            print("Client Declared")
            blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob="24_million/"+filename)
            # file_path=os.path.join(self.upload_path,filename)
            print("Satrting Upload")
            with open(file_path, "rb") as data:
                blob_client.upload_blob(data)
        except:
            print("Upload Failed")

    def upload_folder(self):
        self.blob_service_client =  BlobServiceClient.from_connection_string(self.connection_string)    
        self.my_container = self.blob_service_client.get_container_client(self.container_name)
        for root, dirs, files in os.walk(self.upload_path):
            for file in files:
                try:
                    
                    blob_client = self.blob_service_client.get_blob_client(container=self.container_name, blob=file)
                    file_path= os.path.join(root,file)
                    with open(file_path, "rb") as data:
                        blob_client.upload_blob(data)
                        print('Upload Success :{}'.format(file))

                except:
                    print('Upload Failed : {}'.format(file))

           
az=AZStorage()
# az.upload_folder()
az.upload_file('test_100.jpg', "./test_100.jpg")
# # az.download_blob('Sorted','jpg')
# az.make_file_list()
# az.download_blob('manual_mask/model/pro-images/','zip')
# az.download_blob('medium_resolution/cars/masks/','jpg')
 