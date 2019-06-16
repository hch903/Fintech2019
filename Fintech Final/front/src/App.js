import React, {Component} from 'react';
import axios from 'axios';
import {
  Button,
  Card,
  CardHeader,
  Container,
  Col,
  Form,
  FormGroup,
  Input,
  Label,
  Row,
} from 'reactstrap';
import './App.css';

const stocks = ['台積電','鴻海','台塑','大立光','聯發科','中華電','中信金','台化','統一','國泰金','南亞'];
const etf = ['0050','0058','00690','00692','00730','00731','3036','3074','006204','006208','EWT','FLTW'];
class App extends Component {
  constructor(props){
    super(props);
    this.handleFormSubmit = this.handleFormSubmit.bind(this);
    this.renderResult = this.renderResult.bind(this);

    this.state = {
      title: '',
      content: '',
      response: [],
    }
  }
  handleFormSubmit = e => {
    e.preventDefault();

    const { title, content } = this.state;

    if( !title || !content ) return;

    const newArticle = {
      title: this.state.title,
      content: this.state.content,
    }
    axios.post("https://etf-select.nctu.me:8000/score/", newArticle)
      .then(res => {
        this.setState({response: res.data.msg})
        console.log(res.data.msg);
      })
      .catch(err => console.log(err));
    // this.renderResult(0);
    
    this.setState({
      title: '',
      content: ''
    })
  }

  renderResult(index) {
    const etfName = etf[index];
    if(this.state.response[etfName] !== undefined){
      return Object.keys(this.state.response[etfName]).map((key, i) => {
        const stock = stocks[key-1];
        const probability = (this.state.response[etfName][key][0]*100).toFixed(2);
        let up_or_down = this.state.response[etfName][key][1];
        if(up_or_down === 1)
          up_or_down = '上漲';
        else
          up_or_down = '下跌';
        const percentage = this.state.response[etfName][key][2];
        if(percentage === 0)return
        else {
          return (
            <div className="content" key={i}>
              <div>{`${stock}(佔比${percentage}%)有${probability}%的${up_or_down}機率`}</div>
              {/* <div>{`${stock}在其中佔了%的持股比例`}</div> */}
            </div>
          )
        }
        
      })
    }
    else{
      return (
        <div className="NaN">NaN</div>
      )
    }
  }
  render() {
    return (
      <Container>
        <Row>
          <Col>
            <h1 className="title">挑選被動型ETF</h1>
          </Col>
        </Row>
        <Row>
          <Col>
            <Form onSubmit={this.handleFormSubmit}>
              <FormGroup row>
                <Label for="articleTitle" sm={1}>新聞標題</Label>
                <Col>
                  <Input 
                    sm={11}
                    name="articleTitle"
                    value={this.state.title}
                    id="articleTitle"
                    placeholder="請貼上新聞標題..."
                    onChange={e => {
                      this.setState({title: e.target.value})
                    }}
                    ></Input>
                </Col>
              </FormGroup>
              <FormGroup row>
                <Label for="articleContent" sm={1}>新聞內容</Label>
                <Col>
                  <Input 
                    sm={11}
                    type="textarea"
                    name="articleContent"
                    value={this.state.content}
                    id="articleContent"
                    placeholder="請貼上新聞內文..."
                    onChange={e => {
                      this.setState({content: e.target.value})
                    }}
                    ></Input>
                </Col>
              </FormGroup>
              <Button type="submit" color="primary">
                送出!
              </Button>
            </Form>    
          </Col>
        </Row>

        <Row>
          <Col sm={3} style={{marginTop: "30px"}}>
            <div className="etf-title">元大台灣卓越50基金</div>
            {this.renderResult(0)}
          </Col>
          <Col sm={3} style={{marginTop: "30px"}}>
            <div className="etf-title">iShares MSCI台灣ETF</div>
            {this.renderResult(1)}
          </Col>
          <Col sm={3} style={{marginTop: "30px"}}>
            <div className="etf-title">富邦臺灣公司治理100基金</div>
            {this.renderResult(2)}
          </Col>
          <Col sm={3} style={{marginTop: "30px"}}>
            <div className="etf-title">富邦道瓊臺灣優質高息30ETF基金</div>
            {this.renderResult(3)}
          </Col>
        </Row>
        <Row>
          <Col sm={3} style={{marginTop: "30px"}}>
            <div className="etf-title">復華富時台灣高股息低波動基金</div>
            {this.renderResult(4)}
          </Col>
          <Col sm={3} style={{marginTop: "30px"}}>
            <div className="etf-title">Xtrackers MSCI 台灣 UCITS ETF</div>
            {this.renderResult(5)}
          </Col>
          <Col sm={3} style={{marginTop: "30px"}}>
            <div className="etf-title">iShares安碩核心MSCI台灣指數ETF</div>
            {this.renderResult(6)}
          </Col>
          <Col sm={3} style={{marginTop: "30px"}}>
            <div className="etf-title">富邦台灣釆吉50基金</div>
            {this.renderResult(7)}
          </Col>
        </Row>
        <Row>
          <Col sm={3} style={{marginTop: "30px"}}>
            <div className="etf-title">兆豐國際臺灣藍籌30ETF基金</div>
            {this.renderResult(8)}
            
          </Col>
          <Col sm={3} style={{marginTop: "30px"}}>
            <div className="etf-title">Franklin FTSE台灣ETF</div>
            {this.renderResult(9)}
          </Col>
          <Col sm={3} style={{marginTop: "30px"}}>
            <div className="etf-title">富邦台灣發達基金</div>
            {this.renderResult(10)}
          </Col>
          <Col sm={3} style={{marginTop: "30px"}}>
            <div className="etf-title">永豐臺灣加權ETF基金(豐臺灣)</div>
            {this.renderResult(11)}
          </Col>
        </Row>
      </Container>
    );
  }
  
}

export default App;
