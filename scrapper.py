import requests
from bs4 import BeautifulSoup
import json

def scrape_stj_sumulas():
    
    
    TARGET_URL = "https://scon.stj.jus.br/SCON/sumstj/toc.jsp?numDocsPagina=700&l=700&situacao=%40DOCN+NAO+CANCELADA.INDE.&b=SUMU&ordenacao=%40NUM"
    
    all_sumulas = []
    
    print("Iniciando a coleta das Súmulas do STJ...")
    
    try:
        print(f"Buscando todos os dados da URL: {TARGET_URL}")
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(TARGET_URL, headers=headers, timeout=60)
        response.raise_for_status()

    except requests.exceptions.RequestException as e:
        print(f"Erro ao acessar a URL: {e}")
        return

    soup = BeautifulSoup(response.content, 'html.parser')
    
    sumula_blocks = soup.find_all('div', class_='gridSumula')
    
    if not sumula_blocks:
        print("Nenhum bloco de súmula com a classe 'gridSumula' foi encontrado. Verifique o HTML do site.")
        return

    for block in sumula_blocks:
        num_span = block.find('span', class_='numeroSumula')
        
        text_div = block.find('div', class_='blocoVerbete')
        
        if num_span and text_div:
            ramo_span = text_div.find('span', class_='ramoSumula')
            if ramo_span:
                ramo_span.extract() 
            
            numero = num_span.get_text(strip=True)
            texto = text_div.get_text(strip=True)
            
            sumula_id = f"Súmula {numero}"
            
            sumula_data = {
                "id": sumula_id,
                "texto": texto
            }
            
            all_sumulas.append(sumula_data)

    if all_sumulas:
        print(f"\nColeta finalizada. Total de {len(all_sumulas)} súmulas encontradas.")
        
        try:
            with open('sumulas.json', 'w', encoding='utf-8') as f:
                json.dump(all_sumulas, f, ensure_ascii=False, indent=4)
            print("Arquivo 'sumulas.json' salvo com sucesso!")
        except IOError as e:
            print(f"Erro ao salvar o arquivo JSON: {e}")
    else:
        print("Nenhuma súmula foi coletada. O arquivo JSON não foi gerado.")

if __name__ == "__main__":
    scrape_stj_sumulas()