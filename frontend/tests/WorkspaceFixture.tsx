import App from "../src/App";
export default function WorkspaceFixture(){
  return <App session={{buyerId:"test-buyer",accessToken:"test-token",expiresAt:Date.now()+3600000}} onLogout={()=>{}}/>;
}
