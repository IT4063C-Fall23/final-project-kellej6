# Climate change's effect on local food and water resources.

## Project Overview

The original idea of this project was to determine if climate change affected my local area of Ripley County, Indiana, and what affect it caused on food production and the area’s water resources, mainly streams.  

As I started collecting data, the idea morphed to determining if there was an effect of climate change relating to the agricultural production of corn, since this area is full of corn fields.

## Self Assessment and Reflection

<!-- Edit the following section with your self assessment and reflection -->

### Self Assessment
<!-- Replace the (...) with your score -->

| Category          | Score    |
| ----------------- | -------- |
| **Setup**         | 10 / 10  |
| **Execution**     | 20 / 20  |
| **Documentation** | 10 / 10  |
| **Presentation**  | 30 / 30  |
| **Total**         | 70 / 70  |

### Reflection
<!-- Edit the following section with your reflection -->

#### What went well?
- Two of the three data sources were easily collected.
- Using the Data Wrangling extension was helpful in determine the usefulness of one data source.
- Once I understood that one of the weather data sources was useless, I was easily able to find additional sources.
- Using the GitHub Copilot extension, I was able to use prompts to help convert the API JSON response into a pandas dataframe. 
#### What did not go well?
- Before the GitHub Copilot extension, I struggled with converting the API JSON response into a pandas dataframe.  The JSON response was not sent in an ordered fashion.  Each array (line) had the correct variables, but they differed in the order within the arrays.  Using the extension, it suggested a series of conversion steps that eventually provided the correct order of variables.
#### What did you learn?
- I learned that even though I can successfully implement a ML model, it would be better to have an adequate amount of data.  I have nine years of agricultural data, 2019 was missing.
- I learned from previous assignments how to impute data, namely missing data for 2019.
- I learned from previous assignments how to generate prediction data, namely data for 2023.
- I learned the conversion steps of converting unordered JSON responses into a correct format and converted it into a pandas dataframe.
- I learned to aggregate data into yearly summaries.
- I learned to combine two dataframes into a single dataframe.
#### What would you do differently next time?
- I think I would add temperature data into the analysis.  Precipitation is not the only factor that matters in corn production.
- I would also isolate the weather data to the "growing" season (May through October).  Yearly average precipitation could be misleading as winter months could account for higher values.
- I would also like to determine if droughts occurred during this ten year span, 2013-2023.
---

## Getting Started
### Installing Dependencies

To ensure that you have all the dependencies installed, and that we can have a reproducible environment, we will be using `pipenv` to manage our dependencies. `pipenv` is a tool that allows us to create a virtual environment for our project, and install all the dependencies we need for our project. This ensures that we can have a reproducible environment, and that we can all run the same code.

```bash
pipenv install
```

This sets up a virtual environment for our project, and installs the following dependencies:

- `ipykernel`
- `jupyter`
- `notebook`
- `black`
  Throughout your analysis and development, you will need to install additional packages. You can can install any package you need using `pipenv install <package-name>`. For example, if you need to install `numpy`, you can do so by running:

```bash
pipenv install numpy
```

This will update update the `Pipfile` and `Pipfile.lock` files, and install the package in your virtual environment.

## Helpful Resources:
* [Markdown Syntax Cheatsheet](https://docs.github.com/en/get-started/writing-on-github/getting-started-with-writing-and-formatting-on-github/basic-writing-and-formatting-syntax)
* [Dataset options](https://it4063c.github.io/guides/datasets)