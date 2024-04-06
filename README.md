# Advanced RAG system on legal data


## Get Legifrance API keys

First you need to sign up to the official french website https://developer.aife.economie.gouv.fr/en that gives you access to many public APIs.

![plot](./data/images/sign_up.png)

After signing up, navigate to Applications and create a new application.

![plot](./data/images/list_applications.png)

On the next page, you need to enter the application name, email, structure information (for which I simply put the description), and the application manager's name (I put mine). After entering these details, save the application.

![plot](./data/images/create_application.png)

Then you will have to consent on the terms of service by clicking on "Click here to access to the consent page".

![plot](./data/images/consent_page.png)

Look for legifrance, tick the box corresponding to PROD environment and finally click validate my ToS choices.

![plot](./data/images/tos.png)

Then go back to the created application, click on Edit Application and scroll to Select APIs. In the corresponding table, find Légifrance and tick on Subscribed . Finally click on Apply changes .

![plot](./data/images/select_api.png)

Now go back to the application main page and look for the OAuth Credentials.

![plot](./data/images/credentials.png)

In `./data_ingestion/preprocess_legifrance_data.py`, you can see how we can set the API keys. The env LEGIFRANCE_API_KEY corresponds to the Client ID above, and LEGIFRANCE_API_SECRET corresponds to the Secret key.

```
from pylegifrance import LegiHandler

client = LegiHandler()
client.set_api_keys(
    legifrance_api_key=os.getenv("LEGIFRANCE_API_KEY"),
    legifrance_api_secret=os.getenv("LEGIFRANCE_API_SECRET"),
)
```
